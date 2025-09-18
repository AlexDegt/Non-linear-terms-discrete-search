import torch

from torch import nn
from torch.nn import functional as F
import torch

class PolicyModel(nn. Module):
    def __init__(self):
        super().__init__()
        self.h = 64

        self.policy_model = < Create your model >

        self.value_model = < Create your model >

    def get_policy(self, x):
        < insert your code here >
        return means, var

    def get_value(self, x):
        out = self.value_model(x.float())
        return out

    def forward(self, x):
        policy = self.get_policy(x)
        value = self.get_value(x)

        return policy, value