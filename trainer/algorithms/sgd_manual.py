import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import sys
sys.path.append('../../')

from utils import Timer
from oracle import Oracle

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_sgd_manual(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, test_dataset: DataLoaderType, 
                     loss_fn: LossFnType, quality_criterion: LossFnType, batch_to_tensors: BatchTensorType, save_path: OptionalStr = None, 
                     exp_name: OptionalStr = None, save_every: OptionalInt = None, save_signals: bool = False, weight_names: StrOrList = None,
                     epochs: OptionalInt = None):
    """
    Function implements Stochastic Gradient Descent. The main difference from the standart approach loss.backward() is that
    computation of Wirtinger gradient is implemented through the calculation of Jacobian of loss function with respect to
    the real and imaginary part of model parameters. Then calculated derivatives are gathered to created Wirtinger derivatives
    with respect to the conjugated complex model parameters. 

    Attention! The result, obtained by loss.backward() approach is differs from the proposed approach!

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
        test_dataset (DataLoader, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
                process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        save_signals (bool): The flag that shows, whether to save training signals or not. Defaults to False.
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".
        epochs (int, optional): The number of algorithm iterations on full training dataset. Defaults to "None".

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    # Algorithm stop criteria parameters
    if epochs is None:
        epochs = int(1e+5)

    if save_every is None:
        save_every = epochs - 1

    print_every = 1

    SICOracle = Oracle(model, loss_fn)

    mu = 1e-10
    beta = 0.9
    cache = 0
    lrs = []
    learning_curve_validat = []
    learning_curve_test = []
    grad_norm_curve = []
    weights_norm_curve = []
    grad_norm = None
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()
    with torch.no_grad():
        for batch in test_dataset:
            loss_val = loss_fn(model, batch)
            criterion_val_test = quality_criterion(model, batch)
            best_criterion = criterion_val_test
            print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f}.".format(loss_val.item(), criterion_val_test))
            break
    for epoch in range(epochs):
        for j, batch in enumerate(train_dataset):
            timer.__enter__()

            if type(loss_val) == torch.Tensor:
                loss_val = loss_val.item()

            grad = SICOracle.gradient(batch, weight_names)
            cache = beta * cache + mu * grad
            direction = -1. * cache      
            x = SICOracle.get_flat_params(name_list=weight_names)
            curr_params = x + direction
            SICOracle.set_flat_params(curr_params, name_list=weight_names)

            lrs.append(mu)
            grad_norm = torch.norm(grad).item()
            grad_norm_curve.append(grad_norm)
            weights_norm_curve.append(torch.norm(curr_params).item())

        with torch.no_grad():

            for test_batch in test_dataset:
                criterion_val_test = quality_criterion(model, test_batch)
            for validate_batch in validate_dataset:
                criterion_val_validat = quality_criterion(model, validate_batch)

            learning_curve_test.append(criterion_val_test)
            learning_curve_validat.append(criterion_val_validat)

            if criterion_val_test < best_criterion:
                best_criterion = criterion_val_test
                torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
            if epoch % save_every == 0:
                np.save(save_path + f'lc_validat{exp_name}.npy', np.array(learning_curve_validat))
                np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
                np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
                np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
                np.save(save_path + f'lrs{exp_name}.npy', np.array(lrs))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, quality_criterion_test = {criterion_val_test:.8f}, " + \
                  f"quality_criterion_validat = {criterion_val_validat:.8f}, stepsize = {lrs[-1]:.6e}, " + \
                  f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}")

        general_timer.__exit__()
        print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion