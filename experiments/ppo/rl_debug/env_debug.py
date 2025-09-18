import sys

sys.path.append('../../../')
import os 

import torch
import random
import numpy as np
from oracle import count_parameters
from utils import dataset_prepare, PerformanceEnv
from scipy.io import loadmat
from model import ParallelCheby2D
from copy import deepcopy

device = "cuda:1"
dtype = torch.complex128

delays=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
model = ParallelCheby2D(order=8, delays=delays, channel='A', dtype=dtype, device=device)

delays_number = sum([len(delay_set) for delay_set in delays])
delays_range = [-15, 15]
max_delay_step = 5
delays2change_num = 2

env = PerformanceEnv(model, delays_number, delays_range, max_delay_step, delays2change_num)

action = env.action_space.sample()
state = env.state_space.sample()

print(f"action sample: {action}")
print(f"state sample: {state}")