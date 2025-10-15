import torch
import torch.nn as nn
import sys

from copy import deepcopy

from .layers import Cheby2D, Delay

def tensor_memory_in_gb(tensor):
    bytes_used = tensor.element_size() * tensor.numel()
    gb_used = bytes_used / (1024 ** 3)
    return gb_used

class ParallelCheby2D(nn.Module):
    """
        Class represent 2-dimensional non-linearity based on Chebyshev polynomials
    """
    def __init__ (self, order, delays, channel='A', dtype=torch.complex128, device='cuda:0', trans_len=0):
        super(ParallelCheby2D, self).__init__()
        
        self.dtype = dtype
        self.device = device
        self.delays = delays
        # Channel to compensate distortion
        self.channel = channel
        # Must be list of 2 ints or 1 int
        self.order = order
        delays_input = [delays_branch[1:] for delays_branch in delays]
        delays_output = [delays_branch[:1] for delays_branch in delays]
        self.delay_inp = Delay(delays_input, dtype, device)
        self.delay_out = Delay(delays_output, dtype, device)
        self.cells = nn.ModuleList()
        self.trans_len = trans_len
        for i in range(len(delays)):
            self.cells.append(Cheby2D(order, dtype, device))
    
    def forward(self, x):
        if self.channel == 'A':
            x_in = self.delay_out(x[:, :1, :])
        elif self.channel == 'B':
            x_in = self.delay_out(x[:, 1:2, :])
        else:
            raise ValueError(f"Parameter channel must be either \'A\' or \'B\', but {self.channel} is given")
        x_curr = self.delay_inp(x)
        output = sum([x_in[:, j_branch, ...] * cell(x_curr[:, j_branch, ...]) for j_branch, cell in enumerate(self.cells)])
        return output[..., self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None]

    def get_jacobian(self, x):
        x_in = self.delay_out(x[:, :1, :])
        x_curr = self.delay_inp(x)

        jacobian = []
        for j_branch, cell in enumerate(self.cells):
            jac_curr = x_in[0, j_branch, 0, ...][:, None] * cell.get_jacobian(x_curr[:, j_branch, ...])
            # jac_curr = torch.diag(x_in[0, j_branch, 0, ...]) @ cell.get_jacobian(x_curr[:, j_branch, ...])
            jac_curr = jac_curr[self.trans_len if self.trans_len > 0 else None: -self.trans_len if self.trans_len > 0 else None, :]
            jacobian.append(jac_curr)
        jacobian = torch.cat(jacobian, dim=1)
        return jacobian

    def get_flat_params(self):
        flat_params = []
        for cell in self.cells:
            flat_params.append(cell.weight.view(-1))
        flat_params = torch.cat(flat_params, dim=0)
        return flat_params

    def set_flat_params(self, flat_params):
        offset = 0
        for j_branch, cell in enumerate(self.cells):
            numel = cell.weight.numel()
            cell.weight.copy_(flat_params[offset:offset + numel].view_as(cell.weight))
            offset += numel

    def set_delays(self, delays):
        """
            Function loads delays to the model.
            Pay attention! Since new delays imply structure modification, then
            model parameters are re-initialized.
        """
        delays_input = [delays_branch[1:] for delays_branch in delays]
        delays_output = [delays_branch[:1] for delays_branch in delays]
        self.delay_inp = Delay(delays_input, self.dtype, self.device)
        self.delay_out = Delay(delays_output, self.dtype, self.device)
        self.delays = deepcopy(delays)
        self.cells = nn.ModuleList()
        for i in range(len(delays)):
            self.cells.append(Cheby2D(self.order, self.dtype, self.device))

    def count_parameters(self, trainable=False):
        if trainable:
            return sum([p.numel() for p in self.parameters() if p.requires_grad == True])
        else:
            return sum([p.numel() for p in self.parameters()])