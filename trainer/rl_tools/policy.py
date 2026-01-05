import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
import sys
import math

def one_hot_features(t: torch.Tensor, T: int):
    return torch.nn.functional.one_hot(t.squeeze(-1).long(), num_classes=T).float()

def fourier_features(t: torch.Tensor, T: int, M: int=32):
    assert M % 2 == 0
    num_freq = M // 2

    t = t.float() / float(T)

    freqs = torch.pow(2.0, torch.arange(num_freq, device=t.device, dtype=t.dtype))
    omega = 2 * math.pi * freqs

    angles = t * omega
    features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    return features

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, device: str):
        super().__init__()
        self.emb_size = emb_size
        self.device = device

    def forward(self, positions):
        # positions: тензор размерности (B, 1)
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, device=self.device)
        
        positions = positions.float().to(self.device)  # (B, 1)
        
        # Создаем индексы для частот
        i = torch.arange(0, self.emb_size, dtype=torch.float, device=self.device)  # (emb_size,)
        
        # Вычисляем частоты для каждого измерения
        frequencies = torch.exp(i * (-math.log(15.0) / self.emb_size))  # (emb_size,)
        
        # Вычисляем углы с использованием broadcasting
        # positions: (B, 1), frequencies: (emb_size,) -> angles: (B, emb_size)
        angles = positions * frequencies.unsqueeze(0)  # (B, emb_size)
        
        # Создаем маску для четных индексов
        even_mask = torch.arange(0, self.emb_size, device=self.device) % 2 == 0  # (emb_size,)
        
        # Применяем sin к четным индексам и cos к нечетным
        # angles: (B, emb_size), even_mask: (emb_size,) -> pe: (B, emb_size)
        pe = torch.where(
            even_mask.unsqueeze(0),  # (1, emb_size) для broadcasting
            torch.sin(angles),
            torch.cos(angles)
        )
        
        return pe

class CNNSharedBackPolicy(nn. Module):
    """
        Shared-back policy model on base of 1d convolutions.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_shared_size=4, hidden_shared_num=2, kernel_shared_size=3,
                 hidden_separ_size=4, hidden_separ_num=2, kernel_separ_size=3,
                 device='cuda'):
        super().__init__()

        self.device = device

        self.layer_inp = nn.Conv1d(in_channels=state_dim, out_channels=hidden_shared_size, kernel_size=kernel_shared_size, stride=1, padding=1, device=device)

        # Shared backbone part of policy model
        self.hidden_shared = torch.nn.ModuleList()
        for _ in range(hidden_shared_num):
            self.hidden_shared.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_shared_size, kernel_size=kernel_shared_size, stride=1, padding=1, device=device))

        # Separate part corresponding to policy head training
        self.hidden_policy = torch.nn.ModuleList()
        for j in range(hidden_separ_num):
            if j == 0:
                self.hidden_policy.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))
            else:
                self.hidden_policy.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))
        
        self.policy_out = torch.nn.ModuleList()
        self.delay_range = torch.nn.ModuleList()
        self.delay_steps = torch.nn.ModuleList()
        for j in range(delays2change_num):
            self.delay_range.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=state_dim, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))
            self.delay_steps.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=delays_steps_num, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))
        self.policy_out.append(self.delay_range)
        self.policy_out.append(self.delay_steps)

        # Separate part corresponding to value head training
        self.hidden_value = torch.nn.ModuleList()
        for j in range(hidden_separ_num):
            if j == 0:
                self.hidden_value.append(nn.Conv1d(in_channels=hidden_shared_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))
            else:
                self.hidden_value.append(nn.Conv1d(in_channels=hidden_separ_size, out_channels=hidden_separ_size, kernel_size=kernel_separ_size, stride=1, padding=1, device=device))

        self.value_out = nn.Conv1d(in_channels=hidden_separ_size, out_channels=1, kernel_size=kernel_separ_size, stride=1, padding=1, device=device)

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
                if len(logits.size()) == 3:
                    # For batched version
                    logits = logits.permute(0, 2, 1)
                elif len(logits.size()) == 2:
                    # For unbatched version
                    logits = logits.permute(1, 0)
                policy.append(torch.distributions.Categorical(logits=logits))
                # print(policy[-1].probs.size())
                # print(policy[-1].probs)
                # print(policy[-1].probs.sum())
                # sys.exit()

        x_value = x_shared.clone()
        for layer in self.hidden_value:
            x_value = layer(x_value)
        value = self.value_out(x_value)
        if len(value.size()) == 3:
            # For batched version
            value = value[:, 0, :]
        elif len(value.size()) == 2:
            # For batched version
            value = value[0, :]

        return policy, value

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum([p.numel() for p in self.parameters() if p.requires_grad == True])
            print(f"Total trainable parameters number {param_num}")
        else:
            param_num = sum([p.numel() for p in self.parameters()])
            print(f"Total parameters number {param_num}")
        return param_num

class MLPSharedBackPolicy(nn.Module):
    """
    Shared-back policy model using fully connected layers instead of Conv1d.
    Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_shared_size=4, hidden_shared_num=2,
                 hidden_separ_size=4, hidden_separ_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num

        # Input layer: state_dim -> hidden_shared_size
        self.layer_inp = nn.Linear(state_dim, hidden_shared_size, device=device)
        # Shared backbone
        self.hidden_shared = nn.ModuleList([
            nn.Linear(hidden_shared_size, hidden_shared_size, device=device)
            for _ in range(hidden_shared_num)
        ])

        # Policy branch
        self.hidden_policy = nn.ModuleList()
        for j in range(hidden_separ_num):
            in_ch = hidden_shared_size if j == 0 else hidden_separ_size
            self.hidden_policy.append(nn.Linear(in_ch, hidden_separ_size, device=device))

        # Policy output
        self.delay_range = nn.ModuleList([
            nn.Linear(hidden_separ_size, state_dim, device=device)
            for _ in range(delays2change_num)
        ])
        self.delay_steps = nn.ModuleList([
            nn.Linear(hidden_separ_size, delays_steps_num, device=device)
            for _ in range(delays2change_num)
        ])
        self.policy_out = [self.delay_range, self.delay_steps]

        # Value branch
        self.hidden_value = nn.ModuleList()
        for j in range(hidden_separ_num):
            in_ch = hidden_shared_size if j == 0 else hidden_separ_size
            self.hidden_value.append(nn.Linear(in_ch, hidden_separ_size, device=device))
        self.value_out = nn.Linear(hidden_separ_size, 1, device=device)

    def forward(self, x):
        # x: (length, channels)
        # Input
        x_shared = torch.tanh(self.layer_inp(x))
        for layer in self.hidden_shared:
            x_shared = torch.tanh(layer(x_shared))

        # Policy branch
        x_policy = x_shared.clone()
        for layer in self.hidden_policy:
            x_policy = torch.tanh(layer(x_policy))

        policy = []
        for p_out in self.policy_out:
            for p in p_out:
                logits = p(x_policy)            # (length, out_dim)
                policy.append(torch.distributions.Categorical(logits=logits))

        # Value branch
        x_value = x_shared.clone()
        for j, layer in enumerate(self.hidden_value):
            x_value = torch.tanh(layer(x_value))
        value = self.value_out(x_value)         # (length, 1)
        value = value.squeeze(-1)               # (length,)

        return policy, value

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class MLPSeparatePolicy(nn.Module):
    """
    Model with separated policy and valueheads using fully connected layers.
    Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_policy_size=4, hidden_policy_num=2,
                 hidden_value_size=4, hidden_value_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num

        self.act = torch.nn.SiLU()

        self.policy = nn.ModuleList()
        self.value = nn.ModuleList()

        # Policy branch
        for j in range(hidden_policy_num):
            in_ch = state_dim if j == 0 else hidden_policy_size
            self.policy.append(nn.Linear(in_ch, hidden_policy_size, device=device))

        # Policy output
        self.delay_range = nn.ModuleList([
            nn.Linear(hidden_policy_size, state_dim, device=device)
            for _ in range(delays2change_num)
        ])
        self.delay_steps = nn.ModuleList([
            nn.Linear(hidden_policy_size, delays_steps_num, device=device)
            for _ in range(delays2change_num)
        ])
        self.policy_out = [self.delay_range, self.delay_steps]

        # Value branch
        for j in range(hidden_value_num):
            in_ch = state_dim if j == 0 else hidden_value_size
            self.value.append(nn.Linear(in_ch, hidden_value_size, device=device))
        self.value_out = nn.Linear(hidden_value_size, 1, device=device)

    def forward(self, x):
        # x: (length, channels)

        # Policy branch
        x_policy = x.clone()
        for layer in self.policy:
            x_policy = self.act(layer(x_policy))

        policy = []
        for p_out in self.policy_out:
            for p in p_out:
                logits = p(x_policy)            # (length, out_dim)
                policy.append(torch.distributions.Categorical(logits=logits))

        # Value branch
        x_value = x.clone()
        for j, layer in enumerate(self.value):
            x_value = self.act(layer(x_value))
        value = self.value_out(x_value)         # (length, 1)
        value = value.squeeze(-1)               # (length,)

        return policy, value

    def policy_parameters(self):
        params = []

        if isinstance(self.policy, nn.Module):
            params += list(self.policy.parameters())
        elif isinstance(self.policy, (nn.ModuleList, nn.Sequential)):
            params += list(self.policy.parameters())

        if isinstance(self.delay_range, nn.Module):
            params += list(self.delay_range.parameters())
        elif isinstance(self.delay_range, (nn.ParameterList, list, tuple)):
            for p in self.delay_range:
                assert isinstance(p, nn.Parameter)
            params += list(self.delay_range)

        if isinstance(self.delay_steps, nn.Module):
            params += list(self.delay_steps.parameters())
        elif isinstance(self.delay_steps, (nn.ParameterList, list, tuple)):
            for p in self.delay_steps:
                assert isinstance(p, nn.Parameter)
            params += list(self.delay_steps)

        # sanity-check
        for p in params:
            assert isinstance(p, nn.Parameter) and p.is_leaf, "policy_parameters: non-leaf detected"
        return params

    def value_parameters(self):
        params = []
        if isinstance(self.value, nn.Module):
            params += list(self.value.parameters())
        elif isinstance(self.value, (nn.ModuleList, nn.Sequential)):
            params += list(self.value.parameters())

        params.append(self.value_out.weight)
        if self.value_out.bias is not None:
            params.append(self.value_out.bias)

        for p in params:
            assert isinstance(p, nn.Parameter) and p.is_leaf, "value_parameters: non-leaf detected"
        return params

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class MLPSepDelaySepStep(nn.Module):
    """
        Single-head policy with separate layers for delays indices and steps.
        Model based on fully connected layers with LayerNorm.
        Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_delay_ind_size=128, hidden_delay_ind_num=2,
                 hidden_delay_step_size=128, hidden_delay_step_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num

        self.act = torch.nn.Tanh()
        # self.act = torch.nn.SiLU()

        # Delay indices part
        self.delay_range = nn.ModuleList()
        self.delay_range_ln = nn.ModuleList()
        for j_ind in range(delays2change_num):
            hidden = nn.ModuleList()
            norms = nn.ModuleList()
            for j in range(hidden_delay_ind_num):
                in_ch = state_dim if j == 0 else hidden_delay_ind_size
                out_ch = state_dim if j == hidden_delay_ind_num - 1 else hidden_delay_ind_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
                if j != hidden_delay_ind_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
            self.delay_range.append(hidden)
            self.delay_range_ln.append(norms)

        # Delay steps part
        self.delay_steps = nn.ModuleList()
        self.delay_steps_ln = nn.ModuleList()
        for j_step in range(delays2change_num):
            hidden = nn.ModuleList()
            norms = nn.ModuleList()
            for j in range(hidden_delay_step_num):
                in_ch = state_dim if j == 0 else hidden_delay_step_size
                out_ch = delays_steps_num if j == hidden_delay_step_num - 1 else hidden_delay_step_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
                if j != hidden_delay_step_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
            self.delay_steps.append(hidden)
            self.delay_steps_ln.append(norms)

        self.policy_out = [self.delay_range, self.delay_steps]

    def forward(self, x):
        # x: (length, channels)
        policy = []
        # delay_range
        for branch, norms in zip(self.delay_range, self.delay_range_ln):
            x_policy = x.clone()
            for j_hidden, layer in enumerate(branch):
                if j_hidden != len(branch) - 1:
                    x_policy = layer(x_policy)
                    x_policy = norms[j_hidden](x_policy)
                    x_policy = self.act(x_policy)
                else:
                    logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))

        # delay_steps
        for branch, norms in zip(self.delay_steps, self.delay_steps_ln):
            x_policy = x.clone()
            for j_hidden, layer in enumerate(branch):
                if j_hidden != len(branch) - 1:
                    x_policy = layer(x_policy)
                    x_policy = norms[j_hidden](x_policy)
                    x_policy = self.act(x_policy)
                else:
                    logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))

        return policy

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class MLPSepDelayStep(nn.Module):
    """
        Single-head policy with separate layers for delays indices and steps 
        and additional shared layers.
        Model based on fully connected layers.
        Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 hidden_shared_size=128, hidden_shared_num=2,
                 hidden_delay_ind_size=128, hidden_delay_ind_num=2,
                 hidden_delay_step_size=128, hidden_delay_step_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num

        self.act = torch.nn.SiLU()

        self.shared = nn.ModuleList()

        # Shared part
        for j in range(hidden_shared_num):
            in_ch = state_dim if j == 0 else hidden_shared_size
            self.shared.append(nn.Linear(in_ch, hidden_shared_size, device=device))

        # Delay indices part
        self.delay_range = nn.ModuleList()
        for j_ind in range(delays2change_num):
            hidden = nn.ModuleList()
            for j in range(hidden_delay_ind_num):
                in_ch = hidden_shared_size if j == 0 else hidden_delay_ind_size
                out_ch = state_dim if j == hidden_delay_ind_num - 1 else hidden_delay_ind_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
            self.delay_range.append(hidden)
        
        # Delay steps part
        self.delay_steps = nn.ModuleList()
        for j_step in range(delays2change_num):
            hidden = nn.ModuleList()
            for j in range(hidden_delay_step_num):
                in_ch = hidden_shared_size if j == 0 else hidden_delay_step_size
                out_ch = delays_steps_num if j == hidden_delay_step_num - 1 else hidden_delay_step_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
            self.delay_steps.append(hidden)

        self.policy_out = [self.delay_range, self.delay_steps]

    def forward(self, x):
        # x: (length, channels)

        # Shared part
        x_shared = x.clone()
        for layer in self.shared:
            x_shared = self.act(layer(x_shared))

        policy = []
        for p_out in self.policy_out:
            for branch in p_out:
                x_policy = x_shared.clone()
                for j_hidden, layer in enumerate(branch):
                    if j_hidden != len(branch) - 1:
                        x_policy = self.act(layer(x_policy)) # (length, out_dim)
                    else:
                        logits = layer(x_policy) # (length, out_dim)
                policy.append(torch.distributions.Categorical(logits=logits))

        return policy

    def mask_logits(self, logits, mask, neg_inf=-1e9):
        return logits.masked(~mask, neg_inf)

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class MLPSepDelaySepStepStepID(nn.Module):
    """
        The same as MLPSepDelaySepStep bu with additional Step ID embedding, 
        which helps model to understend index of time step.
        Model based on fully connected layers with LayerNorm.
        Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 trajectory_len, stepid_embed_size,
                 hidden_delay_ind_size=128, hidden_delay_ind_num=2,
                 hidden_delay_step_size=128, hidden_delay_step_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num
        self.stepid_embed_size = stepid_embed_size

        # self.act = torch.nn.Tanh()
        self.act = torch.nn.SiLU()

        self.stepid_embed = torch.nn.Embedding(trajectory_len, stepid_embed_size, device=device)
        # self.stepid_embed = PositionalEncoding(stepid_embed_size, device=device)

        # Delay indices part
        self.delay_range = nn.ModuleList()
        self.delay_range_ln = nn.ModuleList()
        for j_ind in range(delays2change_num):
            hidden = nn.ModuleList()
            norms = nn.ModuleList()
            for j in range(hidden_delay_ind_num):
                in_ch = state_dim + stepid_embed_size if j == 0 else hidden_delay_ind_size
                # in_ch = state_dim + 1 if j == 0 else hidden_delay_ind_size
                out_ch = state_dim if j == hidden_delay_ind_num - 1 else hidden_delay_ind_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
                if j != hidden_delay_ind_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
            self.delay_range.append(hidden)
            self.delay_range_ln.append(norms)

        # Delay steps part
        self.delay_steps = nn.ModuleList()
        self.delay_steps_ln = nn.ModuleList()
        for j_step in range(delays2change_num):
            hidden = nn.ModuleList()
            norms = nn.ModuleList()
            for j in range(hidden_delay_step_num):
                in_ch = state_dim + stepid_embed_size if j == 0 else hidden_delay_step_size
                # in_ch = state_dim + 1 if j == 0 else hidden_delay_step_size
                out_ch = delays_steps_num if j == hidden_delay_step_num - 1 else hidden_delay_step_size
                hidden.append(nn.Linear(in_ch, out_ch, device=device))
                if j != hidden_delay_step_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
            self.delay_steps.append(hidden)
            self.delay_steps_ln.append(norms)

        self.policy_out = [self.delay_range, self.delay_steps]

    def forward(self, x):
        # t_step: (1, length)
        t_step = torch.permute(x["time"], (1, 0)).to(torch.int32)
        # x: (length, channels)
        x = x["state"]
        
        # step_emb = one_hot_features(t=t_step, T=15)
        # step_emb = fourier_features(t=t_step, T=15, M=self.stepid_embed_size)
        # with torch.no_grad():
        #     step_emb = self.stepid_embed(t_step)
        # x = torch.cat((x, step_emb), dim=-1)
        x = torch.cat((x, self.stepid_embed(t_step).squeeze(1)), dim=-1)
        # x = torch.cat((x, t_step / 30), dim=-1)
        policy = []
        # delay_range
        for branch, norms in zip(self.delay_range, self.delay_range_ln):
            x_policy = x.clone()
            for j_hidden, layer in enumerate(branch):
                if j_hidden != len(branch) - 1:
                    x_policy = layer(x_policy)
                    x_policy = norms[j_hidden](x_policy)
                    x_policy = self.act(x_policy)
                else:
                    logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))

        # delay_steps
        for branch, norms in zip(self.delay_steps, self.delay_steps_ln):
            x_policy = x.clone()
            for j_hidden, layer in enumerate(branch):
                if j_hidden != len(branch) - 1:
                    x_policy = layer(x_policy)
                    x_policy = norms[j_hidden](x_policy)
                    x_policy = self.act(x_policy)
                else:
                    logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))

        return policy

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class LSTMShared(nn.Module):
    """
        Policy with shared LSTM for index and step heads. 
        LSTM output is divided into 2 MLP heads: for index and for step.
        Model based on fully connected layers with LayerNorm.
        Input is always (channels, length) without batch dimension.
    """
    def __init__(self, state_dim, delays2change_num, delays_steps_num,
                 trajectory_len, stepid_embed_size,
                 hidden_size=64,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.delays2change_num = delays2change_num
        self.delays_steps_num = delays_steps_num
        self.hidden_size = hidden_size
        self.num_lstm_layers = 1

        # self.act = torch.nn.Tanh()
        # self.act = torch.nn.SiLU()

        self.stepid_embed = torch.nn.Embedding(trajectory_len, stepid_embed_size, device=device)

        self.delay_range = nn.ModuleList()
        self.delay_steps = nn.ModuleList()
        # Shared part (LSTM-based)
        input_size = state_dim + stepid_embed_size
        self.shared_back = nn.LSTM(input_size, hidden_size, num_layers=self.num_lstm_layers, batch_first=True, device=device)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, device=device)
        for _ in range(delays2change_num):
            # Delay indices part (MLP-based)
            self.delay_range.append(nn.Linear(hidden_size, state_dim, device=device))
            # Delay steps part (MLP-based)
            self.delay_steps.append(nn.Linear(hidden_size, delays_steps_num, device=device))

        self.policy_out = [self.delay_range, self.delay_steps]


    def forward(self, x, h, c):
        # t_step: (batch_size, length, 1)
        t_step = x["time"].to(torch.int32)
        x = x["state"]
        x = torch.cat((x, self.stepid_embed(t_step).squeeze(1)), dim=-1)
        policy = []

        # Debug train model of agent. It crushes here!
        x, (h, c) = self.shared_back(x, (h, c))

        x_policy = x.clone()
        for layer in self.delay_range:
            logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))
        for layer in self.delay_steps:
            logits = layer(x_policy)
            policy.append(torch.distributions.Categorical(logits=logits))

        # policy = [distr(delay_0), ..., distr(delay_{N-1}), distr(step_0), ..., distr(step_{M-1})] 
        return policy, h, c

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class MLPConditionalStep(nn.Module):
    def __init__(self, state_dim, delays2change_num, delays_steps_num, trajectory_len,
                 stepid_embed_size=12, ind_choice_embed_size=8,
                 hidden_shared_size=128, hidden_shared_num=2,
                 hidden_delay_ind_size=128, hidden_delay_ind_num=2,
                 hidden_delay_step_size=128, hidden_delay_step_num=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.J = delays2change_num
        self.steps_num = delays_steps_num

        self.act = nn.Tanh()
        # self.act = nn.SiLU()
        self.stepid_embed = nn.Embedding(trajectory_len, stepid_embed_size, device=device)

        # Embedding of chosen delay index
        self.ind_choice_embed = nn.Embedding(state_dim, ind_choice_embed_size, device=device)

        # Shared trunk: in = state + step_id_emb -> hidden_shared_size
        self.shared = nn.ModuleList()
        self.shared_norms = nn.ModuleList()
        in_ch = state_dim + stepid_embed_size
        for l in range(hidden_shared_num):
            out_ch = hidden_shared_size
            self.shared.append(nn.Linear(in_ch, out_ch, device=device))
            if l < hidden_shared_num - 1:
                self.shared_norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
            in_ch = out_ch

        # Index part (each is fed by x_shared, i.e. in_ch = hidden_shared_size)
        self.delay_range = nn.ModuleList()
        self.delay_range_ln = nn.ModuleList()
        for _ in range(self.J):
            layers, norms = nn.ModuleList(), nn.ModuleList()
            in_ch = hidden_shared_size
            for l in range(hidden_delay_ind_num):
                out_ch = self.state_dim if l == hidden_delay_ind_num - 1 else hidden_delay_ind_size
                layers.append(nn.Linear(in_ch, out_ch, device=device))
                if l < hidden_delay_ind_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
                in_ch = out_ch
            self.delay_range.append(layers)
            self.delay_range_ln.append(norms)

        # Step part (conditional w.r.t. chosen index; in_ch = hidden_shared_size + ind_emb)
        self.delay_steps = nn.ModuleList()
        self.delay_steps_ln = nn.ModuleList()
        for _ in range(self.J):
            layers, norms = nn.ModuleList(), nn.ModuleList()
            in_ch = hidden_shared_size + ind_choice_embed_size
            for l in range(hidden_delay_step_num):
                out_ch = self.steps_num if l == hidden_delay_step_num - 1 else hidden_delay_step_size
                layers.append(nn.Linear(in_ch, out_ch, device=device))
                if l < hidden_delay_step_num - 1:
                    norms.append(nn.LayerNorm(out_ch, elementwise_affine=True, device=device))
                in_ch = out_ch
            self.delay_steps.append(layers)
            self.delay_steps_ln.append(norms)

    def forward(self, x):
        """
        x["state"]: (L, state_dim)
        x["time"]:  (L,) or (L,1) integer step_id
        Returns:
          actions:      (J, L, 2) long
          log_probs:    (J, L, 2) float
          dists:        list of length J, each includes [dist_idx, dist_step]
        """
        # ----- time embedding -----
        t = x["time"]
        if t.dim() == 2:  # (L,1) -> (L,)
            t = t.squeeze(-1)
        t = t.to(torch.long)                       # Embedding waits long
        se = self.stepid_embed(t)
        if se.dim() == 3:                      # (1, L, E_step) -> (L, E_step)
            se = se.squeeze(0)                  # (L, E_step)
        st = x["state"]                            # (L, D)
        h = torch.cat([st, se], dim=-1)            # (L, D+E_step)

        # ----- shared trunk -----
        x_shared = h
        for l, layer in enumerate(self.shared):
            x_shared = layer(x_shared)
            if l < len(self.shared) - 1:
                x_shared = self.shared_norms[l](x_shared)
                x_shared = self.act(x_shared)
        # x_shared: (L, hidden_shared_size)

        policy_ind, policy_step = [], []
        chosen_idx, chosen_step = [], []
        logp_idx, logp_step = [], []

        # ----- Index part: sampling i_j -----
        for j in range(self.J):
            x_pol = x_shared.clone()
            for l, layer in enumerate(self.delay_range[j]):
                x_pol = layer(x_pol)
                if l < len(self.delay_range[j]) - 1:
                    x_pol = self.delay_range_ln[j][l](x_pol)
                    x_pol = self.act(x_pol)
            dist_i = Categorical(logits=x_pol)     # (L, state_dim)
            i = dist_i.sample()                    # (L,) long
            policy_ind.append(dist_i)
            chosen_idx.append(i)
            logp_idx.append(dist_i.log_prob(i))    # (L,)

        # ----- Step part: conditional w.r.t. i_j -----
        for j in range(self.J):
            idx_emb = self.ind_choice_embed(chosen_idx[j])   # (L, E_ind)
            x_pol = torch.cat([x_shared, idx_emb], dim=-1)   # (L, hidden+E_ind)
            for l, layer in enumerate(self.delay_steps[j]):
                x_pol = layer(x_pol)
                if l < len(self.delay_steps[j]) - 1:
                    x_pol = self.delay_steps_ln[j][l](x_pol)
                    x_pol = self.act(x_pol)
            dist_k = Categorical(logits=x_pol)     # (L, steps_num)
            k = dist_k.sample()                    # (L,) long
            policy_step.append(dist_k)
            chosen_step.append(k)
            logp_step.append(dist_k.log_prob(k))   # (L,)

        # ----- Gather outputs into tensors -----
        chosen_idx  = torch.stack(chosen_idx,  dim=0)        # (J, L)
        chosen_step = torch.stack(chosen_step, dim=0)        # (J, L)
        logp_idx    = torch.stack(logp_idx,    dim=0).float()# (J, L)
        logp_step   = torch.stack(logp_step,   dim=0).float()# (J, L)

        actions   = torch.stack([chosen_idx, chosen_step], dim=-1)   # (J, L, 2)
        log_probs = torch.stack([logp_idx,   logp_step],   dim=-1)   # (J, L, 2)

        dists = [[policy_ind[j], policy_step[j]] for j in range(self.J)]
        return actions, log_probs, dists

    def count_parameters(self, trainable=False):
        if trainable:
            param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total agent trainable parameters number {param_num}")
        else:
            param_num = sum(p.numel() for p in self.parameters())
            print(f"Total agent parameters number {param_num}")
        return param_num

    def enumerate_parameters(self, trainable=False):
        if trainable:
            params = [(name, p.data.size()) for name, p in self.named_parameters() if p.requires_grad]
            print(f"Agent trainable parameters {params}")
        else:
            params = [(name, p.data.size()) for name, p in self.named_parameters()]
            print(f"Agent parameters {params}")

class Policy:
    def __init__(self, agent):
        self.agent = agent
    
    def reset(self):
        pass

    def act(self, inputs, training=False):
        inputs = torch.tensor(inputs)
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(0)
        # else:
        #     raise ValueError(f"Current RL implementation implies unbatched states. Current state includes {inputs.ndim} dimensions.")
        inputs = inputs.to(self.agent.device).to(torch.float32)

        policy, value = self.agent(inputs)

        indices = policy[:int(len(policy) // 2)]
        steps = policy[int(len(policy) // 2):]

        actions, distr, log_probs = [], [], []
        for j in range(int(len(policy) // 2)):
            distr.append([indices[j], steps[j]])
            indices_ind_j, steps_ind_j = indices[j].sample(), steps[j].sample()
            # steps_j = steps_ind_j - delta_step
            action = torch.cat([indices_ind_j[:, None], steps_ind_j[:, None]], dim=-1)
            log_prob_ind = distr[-1][0].log_prob(indices_ind_j)
            log_prob_step = distr[-1][1].log_prob(steps_ind_j)
            log_prob = torch.cat([log_prob_ind[:, None], log_prob_step[:, None]], dim=-1)
            actions.append(action.detach().cpu().numpy().tolist())
            log_probs.append(log_prob.detach().cpu().numpy().tolist())

        actions = np.array(actions)[:, 0, :].tolist()
        log_probs = np.array(log_probs)[:, 0, :].tolist()

        if not training:
            return {'actions': actions, 
                    'log_probs': log_probs,
                    'values': value.detach().cpu().numpy().tolist()}
        else:
            return {'distribution': distr, 'values': value}

class PolicyActor:
    def __init__(self, agent):
        self.agent = agent
    
    def reset(self):
        pass

    def act(self, inputs, training=False):
        for key, val in inputs.items():
            val = torch.tensor(val)
            if val.ndim < 2:
                val = val.unsqueeze(0)
            # else:
            #     raise ValueError(f"Current RL implementation implies unbatched states. Current state includes {inputs.ndim} dimensions.")
            val = val.to(self.agent.device).to(torch.float32)
            inputs[key] = val

        # print(inputs["state"].shape)
        # sys.exit()

        policy = self.agent(inputs)

        indices = policy[:int(len(policy) // 2)]
        steps = policy[int(len(policy) // 2):]

        actions, distr, log_probs = [], [], []
        for j in range(int(len(policy) // 2)):
            distr.append([indices[j], steps[j]])
            indices_ind_j, steps_ind_j = indices[j].sample(), steps[j].sample()
            # steps_j = steps_ind_j - delta_step
            action = torch.cat([indices_ind_j[:, None], steps_ind_j[:, None]], dim=-1)
            log_prob_ind = distr[-1][0].log_prob(indices_ind_j)
            log_prob_step = distr[-1][1].log_prob(steps_ind_j)
            log_prob = torch.cat([log_prob_ind[:, None], log_prob_step[:, None]], dim=-1)
            actions.append(action.detach().cpu().numpy().tolist())
            log_probs.append(log_prob.detach().cpu().numpy().tolist())

        # print(np.array(actions).shape)
        # print(np.array(log_probs).shape)
        # sys.exit()

        actions = np.array(actions)[:, 0, :].tolist()
        log_probs = np.array(log_probs)[:, 0, :].tolist()

        if not training:
            return {'actions': actions, 
                    'log_probs': log_probs}
        else:
            return {'distribution': distr}

class Policy_v1_3:
    def __init__(self, agent):
        self.agent = agent
    
    def reset(self):
        pass

    def act(self, inputs, training=False):
        # if not training:
        for key, val in inputs.items():
            val = torch.tensor(val)
            if val.ndim < 2:
                val = val.unsqueeze(0)
            val = val.to(self.agent.device).to(torch.float32)
            inputs[key] = val

        actions, log_probs, distr = self.agent(inputs)

        actions = actions[:, 0, :].detach().cpu().numpy().tolist()
        log_probs = log_probs[:, 0, :].detach().cpu().numpy().tolist()

        if not training:
            return {'actions': actions, 
                    'log_probs': log_probs}
        else:
            return {'distribution': distr}

class PolicyMemory:
    def __init__(self, agent):
        self.agent = agent
        # LSTM hidden vectors
        self.h = None
        self.c = None
    
    def reset(self):
        pass

    def act(self, inputs, training=False):

        for key, val in inputs.items():
            val = torch.tensor(val)
            # Number of input tensor dimensions must equal 2:
            # 1-st - sequence length, 2-nd - state dimension.
            if val.ndim == 1:
                val = val.unsqueeze(0)
            elif val.ndim == 0 or val.ndim > 3:
                raise ValueError(f"Number of state dimensions is unacceptable.")
            # else:
            #     raise ValueError(f"Current RL implementation implies unbatched states. Current state includes {inputs.ndim} dimensions.")
            val = val.to(self.agent.device).to(torch.float32)
            inputs[key] = val

        is_first_batch = 0 in inputs["time"]

        if training == False and is_first_batch:
            self.h = torch.zeros((self.agent.num_lstm_layers, self.agent.hidden_size), device=self.agent.device)
            self.c = torch.zeros_like(self.h)
        elif training == True and is_first_batch:
            # Batch size equals the number of trajectories in batch
            batch_size = inputs["state"].size(dim=0)
            self.h = torch.zeros((self.agent.num_lstm_layers, batch_size, self.agent.hidden_size), device=self.agent.device)
            self.c = torch.zeros_like(self.h)

        policy, self.h, self.c = self.agent(inputs, self.h, self.c)

        indices = policy[:int(len(policy) // 2)]
        steps = policy[int(len(policy) // 2):]

        actions, distr, log_probs = [], [], []
        for j in range(int(len(policy) // 2)):
            distr.append([indices[j], steps[j]])
            indices_ind_j, steps_ind_j = indices[j].sample(), steps[j].sample()
            # steps_j = steps_ind_j - delta_step
            action = torch.cat([indices_ind_j[:, None], steps_ind_j[:, None]], dim=-1)
            log_prob_ind = distr[-1][0].log_prob(indices_ind_j)
            log_prob_step = distr[-1][1].log_prob(steps_ind_j)
            log_prob = torch.cat([log_prob_ind[:, None], log_prob_step[:, None]], dim=-1)
            actions.append(action.detach().cpu().numpy().tolist())
            log_probs.append(log_prob.detach().cpu().numpy().tolist())

        # Here actions and log_probs shapes correspond to following:
        # (number of delays to change, sequence length, actions indices)

        # This only used at sampling process.
        # Note that at sampling process always sequence length == 1
        actions = np.array(actions)[:, 0, :].tolist()
        log_probs = np.array(log_probs)[:, 0, :].tolist()

        if not training:
            return {'actions': actions, 
                    'log_probs': log_probs}
        else:
            return {'distribution': distr}