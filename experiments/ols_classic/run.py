import sys

sys.path.append('../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train_ols_classic
from utils import dataset_prepare
from scipy.io import loadmat
from model import ParallelCheby2D
from copy import deepcopy
import yaml
import shutil

# Load yaml file
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Simulation parameters
param_num = config["param_num"]
delays_range = config["delays_range"]
delays_range[1] += 1
iter_num = config["iter_num"]
batch_size = config["batch_size"]
chunk_num = config["chunk_num"]

# The number of signal slots in dataset
slot_num = 1

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
save_path = os.path.join(curr_path, add_folder, exp_name)
os.makedirs(save_path, exist_ok=config["overwrite_file"])

# Save config file in experiment folder
shutil.copyfile(config_path, os.path.join(save_path, "config.yaml"))

# Model initializations
order = param_num
# Define data type
dtype = getattr(torch, config["dtype"])
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
delay_d = 0

# Size of blocks to divide whole signal into
block_size = config["block_size"]
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Flag, which shows whether to return reference signal or not
return_ref = True
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
trans_len = max(abs(np.arange(*delays_range)))
pad_zeros = trans_len
# Channel to compensate: A or B
channel = config["channel"]
dataset, ref_signal = dataset_prepare(data_path, dtype, device, batch_size, block_size, slot_num, pad_zeros, 
                    delay_d, train_slots_ind, validat_slots_ind, test_slots_ind, channel=channel, return_ref=return_ref) 

train_dataset, validate_dataset, test_dataset = dataset

# # Show sizes of batches in train dataset, size of validation and test dataset
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

def tensors_to_batch(batch, x, d):
    batch[0] = x.clone()
    batch[1] = d.clone()
    return batch

def complex_mse_loss(d, y, model):
    error = (d - y)#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
    return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y = batch_to_tensors(signal_batch)
    return complex_mse_loss(y, model(x), model)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
# To avoid conflicts for classification task you can write:
# def quality_criterion(loss_val):
#     return loss_val
def quality_criterion(model, dataset):
    targ_pow, loss_val = 0, 0
    for batch in dataset:
        # _, d= batch_to_tensors(batch)
        # targ_pow += d.abs().square().sum()
        # targ_pow += d[..., trans_len if trans_len > 0 else None: -pad_zeros if pad_zeros > 0 else None].abs().square().sum()
        input_pow = np.sum(np.abs(ref_signal) ** 2)
        loss_val += loss(model, batch)
    # return 10.0 * torch.log10((loss_val) / (targ_pow)).item()
    return 10.0 * torch.log10((loss_val) / (input_pow)).item()

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

# Initialize delays with zeros, since they would be chosen later
delays = [[0, 0, 0]]
model = ParallelCheby2D(order, delays, channel, dtype, device, trans_len)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")

# param_names = [name for name, p in model.named_parameters()]
# params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
# print(params)

best_delays = train_ols_classic(model, train_dataset, train_dataset, train_dataset, loss, quality_criterion, config, batch_to_tensors,
                                        tensors_to_batch, chunk_num=chunk_num, save_path=save_path, exp_name=exp_name,
                                        weight_names=weight_names, delays_range=delays_range, iter_num=iter_num)