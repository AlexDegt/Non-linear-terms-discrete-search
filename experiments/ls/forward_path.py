import sys

sys.path.append('../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train
from utils import dataset_prepare
from scipy.io import loadmat
from scipy.signal import get_window, welch
from model import ParallelCheby2D
from copy import deepcopy
import yaml
import shutil
import pickle

# Load yaml file
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Simulation parameters
param_num = config["param_num"]
delays = config["delays"]
batch_size = config["batch_size"]
chunk_num = config["chunk_num"] # 31 * 18

# The number of signal slots in dataset
slot_num = 2

# For ACLR calculation
f = 10 # MHz
fs = 245.76 # MHz
nfft = 512

device = config["device"]
seed = 964
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data
data_path = config["data_path"]

# Determine experiment name and create its directory
exp_name = config["trial_name"]
add_folder = ''

curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)

# Model initialization
order = param_num
delays = config["delays"]
# Define data type
# dtype = torch.complex64
dtype = torch.complex128
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1, 2), range(1, 2)
delay_d = 0

# Size of blocks to divide whole signal into
block_size = config["block_size"]
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
trans_len = int(len(delays) // 2)
pad_zeros = trans_len
# Channel to compensate: A or B
channel = config["channel"]
dataset = dataset_prepare(data_path, dtype, device, batch_size, block_size, slot_num, pad_zeros, 
                    delay_d, train_slots_ind, validat_slots_ind, test_slots_ind, channel=channel)

train_dataset, validate_dataset, test_dataset = dataset

# Show sizes of batches in train dataset, size of validation and test dataset
# for i in range(len(dataset)):
#     for j, batch in enumerate(dataset[i]):
#         # if j == 0:
#         # Input batch size
#         print(batch[0].size())
#         # Target batch size
#         print(batch[1].size())
#     print(j + 1)
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1][:, :1, :]
    return x, d

def complex_mse_loss(d, y, model):
    # d = d[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
    error = (d - y)
    return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y = batch_to_tensors(signal_batch)
    return complex_mse_loss(y, model(x), model)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
def nmse_fn(model, dataset):
    targ_pow, loss_val = 0, 0
    for batch in dataset:
        _, d = batch_to_tensors(batch)
        targ_pow += d.abs().square().sum()
        loss_val += loss(model, batch)
    return 10.0 * torch.log10((loss_val) / (targ_pow)).item()

def aclr_fn_torch(sig, f, fs=1.0, nfft=1024, window='blackman', nperseg=None, noverlap=None):
    """
    Calculate Adjacent Channel Leakage Ratio using PyTorch.
    
    Parameters:
        sig (torch.Tensor): Complex signal (1D Tensor).
        f (float): Frequency scalar.
        fs (float, optional): Sampling frequency. Defaults to 1.0.
        nfft (int, optional): Number of FFT points. Defaults to 1024.
        window (str, optional): Window type (e.g., 'blackman'). Defaults to 'blackman'.
        nperseg (int, optional): Segment length for Welch's method. Defaults to None.
        noverlap (int, optional): Number of overlapping points. Defaults to None.
    
    Returns:
        float: ACLR in dB.
    """
    if nperseg is None:
        nperseg = nfft  # Default segment length matches FFT size
    if noverlap is None:
        noverlap = nperseg // 2  # Default overlap is 50%

    # Convert the window to a torch tensor
    win = torch.tensor(get_window(window, nperseg, fftbins=True), dtype=torch.float32).to(sig.device)

    # Reshape the signal into overlapping segments
    step = nperseg - noverlap
    num_segments = (len(sig) - noverlap) // step
    segments = torch.stack([
        sig[i * step:i * step + nperseg] for i in range(num_segments)
    ])

    # Apply the window and compute the FFT
    segments = segments * win  # Broadcasting window to each segment
    fft_segments = torch.fft.fft(segments, nfft, dim=-1)
    psd = torch.mean(torch.abs(fft_segments) ** 2, dim=0)  # Welch PSD estimate

    # FFT shift
    psd = torch.fft.fftshift(psd)
    freqs = torch.fft.fftshift(torch.fft.fftfreq(nfft, d=1/fs))
    
    # Compute the indices for the frequency range
    ind1 = (nfft // 2) - int(torch.ceil(torch.tensor(nfft * f / fs)))
    ind2 = (nfft // 2) + int(torch.ceil(torch.tensor(nfft * f / fs)))
    guard = 4  # Guard band size
    
    # Calculate ACLR
    numerator = torch.sum(psd[ind2 + guard: 2 * ind2 - ind1 + guard])
    denominator = torch.sum(psd[ind1: ind2])
    aclr = 10 * torch.log10(numerator / denominator)
    
    return aclr.item()  # Return as a scalar

def load_weights(path_name, device=device):
    return torch.load(path_name, map_location=torch.device(device))

def set_weights(model, weights):
    model.load_state_dict(weights)

def get_nested_attr(module, names):
    for i in range(len(names)):
        module = getattr(module, names[i], None)
        if module is None:
            return
    return module

model = ParallelCheby2D(order, delays, channel, dtype, device)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")
# param_names = [name for name, p in model.named_parameters()]
# params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
# print(params)

set_weights(model, load_weights(os.path.join(load_path, r'weights_best.pt')))

model.eval()
# Type of data to carry out model evaluation on
model_eval_types = ["train", "test"]
with torch.no_grad():
    dataset_all = train_dataset, test_dataset
    for j_dataset, dataset in enumerate(dataset_all):

        NMSE = nmse_fn(model, dataset)
        print(f"NMSE on whole {model_eval_types[j_dataset]} dataset is {NMSE:.3f} dB")
        y, d, x = [], [], []
        for j, batch in enumerate(dataset):
            data = batch_to_tensors(batch)

            y.append(model(data[0])[0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
            d.append(data[1][0, 0, :])#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])
            x.append(data[0][0, 0, trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None])

        y_full_tensor = torch.cat(y, dim=-1).to(device)
        d_full_tensor = torch.cat(d, dim=-1).to(device)
        x_full_tensor = torch.cat(x, dim=-1).to(device)
        y_full_numpy = y_full_tensor.detach().cpu().numpy()
        d_full_numpy = d_full_tensor.detach().cpu().numpy()
        x_full_numpy = x_full_tensor.detach().cpu().numpy()

        aclr_val = aclr_fn_torch((x_full_tensor + d_full_tensor - y_full_tensor)[trans_len:-trans_len], f=f, fs=fs, nfft=nfft)
        aclr_val_dpdoff = aclr_fn_torch((x_full_tensor + d_full_tensor)[trans_len:-trans_len], f=f, fs=fs, nfft=nfft)

        if j_dataset == 0:
            np.save(os.path.join(load_path, r'y.npy'), y_full_numpy)
            np.save(os.path.join(load_path, r'd.npy'), d_full_numpy)
            np.save(os.path.join(load_path, r'x.npy'), x_full_numpy)
        elif j_dataset == 1:
            np.save(os.path.join(load_path, r'y_test.npy'), y_full_numpy)
            np.save(os.path.join(load_path, r'd_test.npy'), d_full_numpy)
            np.save(os.path.join(load_path, r'x_test.npy'), x_full_numpy)