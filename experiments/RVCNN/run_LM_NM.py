import sys

sys.path.append('../../')
import os 

import torch
import random
import yaml
import numpy as np
from oracle import count_parameters
from trainer import train
from utils import dataset_prepare
from scipy.io import loadmat
from model import RVCNN

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# The number of algorithm iterations on full dataset
epochs = int(1.5e+3)

seed_0 = 964
for exp in range(0, 5):
    seed = seed_0 + exp
    # Determine experiment name and create its directory
    exp_name = f"reproduced_paper_exp_{exp}_seed_{seed}_cubic_newton_4_channels_6_5_5_2_ker_size_3_3_3_3_act_sigmoid_1500_epochs"

    add_folder = os.path.join("reproduced_results")
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, add_folder, exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    device = config["device"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch.use_deterministic_algorithms(True)
    if device != "cpu":
        torch.backends.cudnn.deterministic = True

    # Load PA input and output data (2 channels for both: input and output)
    mat = loadmat("../../data/data2d.mat")

    # Define data type
    # dtype = torch.complex64
    dtype = torch.complex128
    # ptype = torch.float32
    ptype = torch.float64

    # Number of output channels of each convolutional layer.
    out_channels = [6, 5, 5, 2]
    # Kernel size for each layer. Kernel sizes must be odd integer numbers.
    # Otherwise input sequence length will be reduced by 1 (for case of mode == 'same' in nn.Conv1d).
    kernel_size = [3, 3, 3, 3]
    # Activation functions are listed in /model/layers/activation.py
    # Don`t forget that model output must be holomorphic w.r.t. model parameters
    activate = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
    p_drop = list(np.zeros_like(out_channels))
    delays = [[0]]
    slot_num = 10
    # Indices of slots which are chosen to be included in train/test set (must be of a range type).
    # Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
    # In full-batch mode train, validation and test dataset are the same.
    # In mini-batch mode validation and test dataset are the same.
    train_slots_ind, validat_slots_ind, test_slots_ind = range(8), range(8), range(8, 10)
    # train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
    delay_d = 0
    # batch_size == None is equal to batch_size = 1.
    # block_size == None is equal to block_size = signal length.
    # Block size is the same as chunk size 
    batch_size = 1
    chunk_num = config["chunk_num"]
    signal_length = 0.8 * 213500
    assert signal_length % chunk_num == 0, \
        f"It is recommended to set number of chunks such that whole signal length {signal_length} is divisible evenly by number of chunks {chunk_num} for correct hessian and gradient accumulation."
    chunk_size = int(signal_length/chunk_num)
    # L2 regularization parameter
    alpha = 0.0
    # Configuration file
    config_train =  None
    # Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
    # Since each 1d convolution in model CVCNN makes zero-padding with int(kernel_size/2) left and right, then 
    # NO additional padding in the input batches is required.
    pad_zeros = 4
    dataset = dataset_prepare(mat, dtype, device, slot_num=slot_num, delay_d=delay_d,
                            train_slots_ind=train_slots_ind, test_slots_ind=test_slots_ind, validat_slots_ind=validat_slots_ind,
                            pad_zeros=pad_zeros, batch_size=batch_size, block_size=chunk_size)

    train_dataset, validate_dataset, test_dataset = dataset

    # Attention here!!! 2 channels of signal can be pre-distorted.
    # In order to pre-distort channel A, desired signal d should be chosen as: d = a[1][:, :1, :].
    # Good performance for channel A is -14.8 - -15.0 dB.
    # In order to pre-distort channel B, desired signal d should be chosen as: d = a[1][:, 1:2, :].
    # Good performance for channel B is -20.5 - -21.0 dB.
    def batch_to_tensors(a):
        x = a[0]
        d = a[1][:, :1, :]
        nf = a[1][:, 2:, :]
        return x, d, nf

    def complex_mse_loss(d, y, model):
        error = (d - y)[..., pad_zeros: -pad_zeros]
        # error = (d - y)
        return error.abs().square().sum() + alpha * sum(torch.norm(p)**2 for p in model.parameters())

    def loss(model, signal_batch):
        x, y, _ = batch_to_tensors(signal_batch)
        return complex_mse_loss(model(x), y, model)
    # This function is used only for telecom task.
    # Calculates NMSE on base of accumulated on every batch loss function
    @torch.no_grad()
    # To avoid conflicts for classification task you can write:
    # def quality_criterion(loss_val):
    #     return loss_val
    def quality_criterion(model, dataset):
        targ_pow, nf_pow, loss_val = 0, 0, 0
        for batch in dataset:
            _, d, nf = batch_to_tensors(batch)
            nf_pow += nf[..., pad_zeros: -pad_zeros].abs().square().sum()
            targ_pow += d[..., pad_zeros: -pad_zeros].abs().square().sum()
            loss_val += loss(model, batch)
        return 10.0 * torch.log10((loss_val - nf_pow) / (targ_pow - nf_pow)).item()

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

    # RVCNN - Real-Valued Convolutional NN.
    # Takes pure signal both channels x_{A, n}, x_{B, n} as an input and 
    # creates input features: Re(x_{A, n}), Im(x_{A, n}), Re(x_{B, n}), Im(x_{B, n}), |x_{A, n}|, |x_{B, n}|. 
    # Thus there're 6 input channels.
    # Output channel numbers are regulated by the list out_channels.
    # Last layer output channels number equal 2, which correspond to Re(x_last_layer) and Im(x_last_layer) part of 
    # pre-distorted signal. Output of the RVCNN is Re(x_last_layer) + 1j * Im(x_last_layer)
    model = RVCNN(device=device, delays=delays, out_channels=out_channels, kernel_size=kernel_size, features=['real', 'imag', 'abs'], 
                activate=activate, batch_norm_mode='nothing', p_drop=p_drop, bias=True, dtype=ptype)

    model.to(device)

    weight_names = list(name for name, _ in model.state_dict().items())

    parameters_num = count_parameters(model)
    print(f"Current model parameters number is {parameters_num}")
    param_names = [name for name, p in model.named_parameters()]
    params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
    # print(params)
    assert parameters_num < chunk_size, \
        f"The number of trainable model parameters {parameters_num} must be lower signal chunk size {chunk_size} for correct work of Newton-based methods."

    def param_init(model):
        branch_num = len(delays)
        for i in range(branch_num):
            for j in range(len(out_channels)):  
                layer_name = f'nonlin.cnn.{j}.conv_layer'.split(sep='.')
                
                layer_module = get_nested_attr(model, layer_name)

                torch.nn.init.normal_(layer_module.weight.data, mean=0, std=1)
                # torch.nn.init.uniform_(layer_module.weight.data, -1, 1)

                # Small initial parameters is important is case of using tanh activation
                layer_module.weight.data *= 1e-2

                if layer_module.bias is not None:
                    torch.nn.init.normal_(layer_module.bias.data, mean=0, std=1)
                    layer_module.bias.data *= 1e-2
        return None

    param_init(model)

    # Define jacobian calculation mode. Can influence calculations speed significantly depending on the model structure.
    # Could be 'forward-mode' or 'reverse-mode'
    jac_calc_strat = 'forward-mode'
    
    # Train type shows which algorithm is used for optimization.
    # train_type='sgd_auto' # gradient-based optimizer.
    # train_type='mnm_damped' # Damped Mixed Newton. Work only with models with complex parameters!
    # train_type='mnm_lev_marq' # Levenberg-Marquardt on base of Mixed Newton. Work only with models with complex parameters!
    # train_type='newton_damped' # Damped Newton. Can be used for models with real and complex parameters.
    train_type='newton_lev_marq' # Levenberg-Marquardt on base of Newton. Can be used for models with real and complex parameters.
    # train_type='cubic_newton' # Cubic Newton. Currently work only with models with complex parameters!
    # train_type='cubic_newton_simple' # Simplified cubic Newton. Currently work only with models with complex parameters!
    learning_curve, best_criterion = train(model, train_dataset, loss, quality_criterion, config_train, batch_to_tensors, validate_dataset, test_dataset, 
                                        train_type=train_type, chunk_num=chunk_num, exp_name=exp_name, save_every=1, save_path=save_path, 
                                        weight_names=weight_names, device=device, jac_calc_strat=jac_calc_strat, epochs=epochs)

    print(f"Best NMSE: {best_criterion} dB")