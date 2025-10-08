import sys

sys.path.append('../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train_ls
from utils import dataset_prepare
from scipy.io import loadmat
from model import ParallelCheby2D
from copy import deepcopy
import yaml
import shutil

from oracle import Oracle

# Load yaml file
config_path = "config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Simulation parameters
param_num = config["param_num"]
delays = config["delays"]
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
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
# NO additional padding in the input batches is required.
trans_len = max([abs(item) for sublist in delays for item in sublist])
pad_zeros = trans_len
# Channel to compensate: A or B
channel = config["channel"]
dataset = dataset_prepare(data_path, dtype, device, batch_size, block_size, slot_num, pad_zeros, 
                    delay_d, train_slots_ind, validat_slots_ind, test_slots_ind, channel=channel) 

train_dataset, validate_dataset, test_dataset = dataset

def batch_to_tensors(a):
    x = a[0]
    d = a[1][:, :1, :]
    return x, d

def complex_mse_loss(d, y, model):
    error = (d - y)#[..., trans_len if trans_len > 0 else None: -trans_len if trans_len > 0 else None]
    return error.abs().square().sum() #+ alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y = batch_to_tensors(signal_batch)
    return complex_mse_loss(y, model(x), model)

model = ParallelCheby2D(order, delays, channel, dtype, device, trans_len)
model.to(device)
weight_names = list(name for name, _ in model.state_dict().items())
print(f"Current model parameters number is {count_parameters(model)}")

# Extract desired
d = []
for j, batch in enumerate(train_dataset):
    data = batch_to_tensors(batch)
    d.append(data[1][0, 0, :])
d = torch.cat(d)

# Calculate jacobians for 2 delay sets separately
SICOracle = Oracle(model, loss)

# First jacobian
SICOracle._model.set_delays([delays[0]])
for j, batch in enumerate(train_dataset):
    jac_0 = SICOracle.jacobian(batch, batch_to_tensors)[0, ...]

# Second jacobian
SICOracle._model.set_delays([delays[1]])
for j, batch in enumerate(train_dataset):
    jac_1 = SICOracle.jacobian(batch, batch_to_tensors)[0, ...]

jac = torch.cat((jac_0, jac_1), dim=1)

# Calculate optimal parameters
c_opt = torch.linalg.pinv(jac.T.conj() @ jac) @ jac.T.conj() @ d

# Calculate error with jacobian created from the same 1-st jac and 2-nd jac
e_simple = d - jac @ c_opt

# Save error
e_simple_numpy = e_simple.detach().cpu().numpy()
np.save(os.path.join(save_path, r'e_simple.npy'), e_simple_numpy)

""" Second approach with orthogonalization of jac_1 w.r.t. jac_0 """
# Orthogonalize jac_1 w.r.t. jac_0
inverse = torch.linalg.pinv(jac_0.T.conj() @ jac_0)
tmp1 = jac_0.T.conj() @ jac_1
tmp2 = jac_0 @ inverse
proj = tmp2 @ tmp1
jac_1_ort = jac_1 - proj

print(f"Sum = {(jac_0.conj().T @ jac_1_ort).abs().sum()}")
print(f"Mean = {(jac_0.conj().T @ jac_1_ort).abs().mean()}")
print(f"Max = {(jac_0.conj().T @ jac_1_ort).abs().max()}")

jac = torch.cat((jac_0, jac_1_ort), dim=1)

# Calculate optimal parameters
c_opt = torch.linalg.pinv(jac.T.conj() @ jac) @ jac.T.conj() @ d

# Calculate error with jacobian created from the same 1-st jac and 2-nd orthogonal to the first jac
e_ort = d - jac @ c_opt

# Save error
e_ort_numpy = e_ort.detach().cpu().numpy()
np.save(os.path.join(save_path, r'e_ort.npy'), e_ort_numpy)

print(f"sum(|e_simple - e_ort|/sum(|e_simple|) = {sum(abs(e_simple_numpy - e_ort_numpy))/sum(abs(e_simple_numpy)):.7f}")

""" Third approach with orthogonalization of jac_1 w.r.t. jac_0 """

c_0 = torch.linalg.pinv(jac_0.T.conj() @ jac_0) @ jac_0.T.conj() @ d
e_0 = d - jac_0 @ c_0
c_1 = torch.linalg.pinv(jac_1_ort.T.conj() @ jac_1_ort) @ jac_1_ort.T.conj() @ e_0
e_ort_1 = e_0 - jac_1_ort @ c_1

# Save error
e_ort_1_numpy = e_ort_1.detach().cpu().numpy()
np.save(os.path.join(save_path, r'e_ort_1.npy'), e_ort_1_numpy)

print(f"sum(|e_simple - e_ort_1|/sum(|e_simple|) = {sum(abs(e_simple_numpy - e_ort_1_numpy))/sum(abs(e_simple_numpy)):.7f}")