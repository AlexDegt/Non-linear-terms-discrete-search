import torch

from torch import nn
from torch.nn import functional as F
import sys

class CNNSharedBackPolicy(nn. Module):
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_shared_size=4, hidden_shared_num=2, kernel_shared_size=3,
                 hidden_separ_size=4, hidden_separ_num=2, kernel_separ_size=3):
        super().__init__()

        self.layer_inp = nn.Conv1d(in_channels=state_dim, out_channels=hidden_shared_size, kernel_size=kernel_shared_size, stride=1, padding=1)

        # Shared backbone part of policy model
        self.hidden_shared = torch.nn.ModuleList()
        for _ in range(hidden_shared_num):
            self.hidden_shared.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_shared_size, kernel_size=kernel_shared_size, stride=1, padding=1))

        # Separate part corresponding to policy head training
        self.hidden_policy = torch.nn.ModuleList()
        for j in range(hidden_separ_num):
            if j == 0:
                self.hidden_policy.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1))
            else:
                self.hidden_policy.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1))
        
        self.policy_out = torch.nn.ModuleList()
        self.delay_range = torch.nn.ModuleList()
        self.delay_steps = torch.nn.ModuleList()
        for j in range(delays2change_num):
            self.delay_range.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=state_dim, kernel_size=kernel_separ_size, stride=1, padding=1))
            self.delay_steps.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=delays_steps_num, kernel_size=kernel_separ_size, stride=1, padding=1))
        self.policy_out.append(self.delay_range)
        self.policy_out.append(self.delay_steps)

        # Separate part corresponding to value head training
        self.hidden_value = torch.nn.ModuleList()
        for j in range(hidden_separ_num):
            if j == 0:
                self.hidden_value.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1))
            else:
                self.hidden_value.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1))

        self.value_out = nn.Conv1d(in_channels=hidden_separ_size, out_channels=1, kernel_size=kernel_separ_size, stride=1, padding=1)

    def forward(self, x):
        x_shared = self.layer_inp(x)

        for layer in self.hidden_shared:
            x_shared = layer(x_shared)

        x_policy = x_shared.clone()
        for layer in self.hidden_policy:
            x_policy = layer(x_policy)

        # Policy is a list of 2 * delays2change_num distributions
        # 1-st delays2change_num distributions correspond to indices of chosen delays to be changed
        # 2-nd delays2change_num distributions correspond to values of delays steps
        policy = []
        for p_out in self.policy_out:
            for p in p_out:
                logits = p(x_policy)
                logits = logits.permute(0, 2, 1)
                policy.append(torch.distributions.Categorical(logits=logits))

        x_value = x_shared.clone()
        for layer in self.hidden_value:
            x_value = layer(x_value)
        value = self.value_out(x_value)
        value = value[:, 0, :]

        return policy, value

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum([p.numel() for p in self.parameters() if p.requires_grad == True])
            print(f"Total trainable parameters number {param_num}")
        else:
            param_num = sum([p.numel() for p in self.parameters()])
            print(f"Total parameters number {param_num}")
        return param_num