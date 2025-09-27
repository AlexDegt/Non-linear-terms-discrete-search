import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import sys

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