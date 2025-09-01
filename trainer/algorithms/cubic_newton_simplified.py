import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np
from copy import copy, deepcopy

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


def update_delta(error, delta_min, delta_max, delta):
    if error > 0:
        delta_max = delta
        delta = (delta_min + delta_max) / 2
    else:
        delta_min = delta
        if np.isinf(delta_max):
            delta *= 2
        else:
            delta = (delta_min + delta_max) / 2
    return delta_max, delta_min, delta

def find_delta(eigenvectors_hess, eigenvalues_hess, lipschitz_constant, eigenvectors_gradient_product, tol=1e-9):
    def get_direction_delta(delta):
        diag_values = eigenvalues_hess + lipschitz_constant * delta / 4
        return -eigenvectors_hess @ (1 / diag_values * eigenvectors_gradient_product)
    tmp = 4 * max(-eigenvalues_hess.min() / lipschitz_constant, 0)
    if isinstance(tmp, torch.Tensor):
        tmp = tmp.cpu().item()
    delta_max, delta_min, delta = np.inf, tmp, 100
    while True:
        direction = get_direction_delta(delta)
        error = delta ** 2 - torch.norm(direction) ** 2
        if abs(error) <= tol or delta_max-delta_min < tol/1e4:
            return delta, direction
        delta_max, delta_min, delta = update_delta(error, delta_min, delta_max, delta)


def train_cubic_newton_simplified(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType,
                              test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
                              batch_to_tensors: BatchTensorType, chunk_num: OptionalInt = None, 
                              save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
                              save_signals: bool = False, weight_names: StrOrList = None, jac_calc_strat: str = "reverse-mode",
                              epochs: OptionalInt = None):
    """
    Function implements damped version of Mixed Newton Method. Mixed Newton implies computation of the mixed Hessian and
    gradient multiplication each algorithm step. Current function uses oracle.Oracle.direction_through_jacobian
    function which firstly accumulates model output jacobian J w.r.t. the model parameters on the whole batch, then 
    calculates Hessian as matrix multiplication (J^H @ J). Gradient is calculated as vector-jacobian multiplication
    (J^H @ e), where e - model error on current batch. 
    
    Attention!
    Mixed Newton performs better on the long batches - with big sample size. Batch size could be chosen as 1.
    For the big sample size (>~ 50000) jacobian matrix requires huge memory resources, that`s why function
    oracle.Oracle.direction_through_jacobian contains chunk-mechanism under the hood. Chunk-mechanism divides
    whole batch into the chunks with chunk_size to save GPU memory.

    Damping of the Mixed Newton is also a key mechanism for the appropriate performance acquirement. It is implemented
    in the following way: firstly inverse Hessian - gradient product is calculated and stored, then model parameters
    updated with current step size. If the quality criterion improved, then step size is increased. If the quality 
    criterion degraded comparing to the previous step, then step size is decreased till the quality criterion 
    is not better comparing to the previous step.

    The stop criteria is determined by gradient norm. If it is lower than min_grad_norm than algorithm stops.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
            Newton-based training methods usually work on the whole signal dataset. 
            Therefore train and validation datasets are implied to be the same.
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
        chunk_num (int, optional): The number chunks in dataset. Defaults to "None".
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
        epochs = int(1.5e+3)

    if save_every is None:
        save_every = epochs - 1

    # Determine jacobian calculation strategy
    if jac_calc_strat == 'forward-mode':
        vectorize = True
    elif jac_calc_strat == 'reverse-mode':
        vectorize = False
    else:
        sys.exit(f'Jacobian calculation strategu must be \'forward-mode\' or \'reverse-mode\', but {jac_calc_strat} is given.')

    epoch, print_every = 0, 1

    SICOracle = Oracle(model, loss_fn)

    mu = 1.
    reg_param_curve = []
    learning_curve_train = []
    learning_curve_test = []
    learning_curve_validate = []
    learning_curve_train_qcrit = []
    learning_curve_test_qcrit = []
    learning_curve_validate_qcrit = []
    grad_norm_curve = []
    weights_norm_curve = []
    grad_norm = None
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()

    def accum_loss(dataset):
        loss_val = 0
        for batch in dataset:
            loss_val += SICOracle.loss_function_val(batch).item()
        return loss_val
            
    # Calculate initial values of loss and quality criterion on validation and test dataset
    with torch.no_grad():
        loss_val_test = accum_loss(test_dataset)
        criterion_val_test = quality_criterion(model, test_dataset)
        best_criterion_test = criterion_val_test
        learning_curve_test.append(loss_val_test)
        learning_curve_test_qcrit.append(criterion_val_test)
        print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f}.".format(loss_val_test, criterion_val_test))
        loss_val_train = accum_loss(train_dataset)
        criterion_val_train = quality_criterion(model, train_dataset)
        learning_curve_train.append(loss_val_train)   
        learning_curve_train_qcrit.append(criterion_val_train)
        print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f}.".format(loss_val_train, criterion_val_train))
        loss_val_validate = accum_loss(validate_dataset)
        criterion_val_validate = quality_criterion(model, validate_dataset)
        learning_curve_validate.append(loss_val_validate)
        learning_curve_validate_qcrit.append(criterion_val_validate)
        print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f}.".format(loss_val_validate, criterion_val_validate))

    epoch = 0
    min_grad_norm = 1e-8    
    for epoch in range(epochs):
    # while grad_norm is None or grad_norm >= min_grad_norm:
        timer.__enter__()
        # Accumulate hessian and gradient on the whole training dataset.
        # Combination of all batches on train dataset should be equal validation dataset
        for j, batch in enumerate(train_dataset):

            # Simplified cubic Newton requires mixed Hessian calculation in contrast to the fair cubic Newton
            delta_hess, delta_grad = SICOracle.direction_through_jacobian(batch, batch_to_tensors, weight_names=weight_names,
                                            strategy=jac_calc_strat, vectorize=vectorize)

            with torch.no_grad():
                if j % chunk_num == 0:
                    hess = torch.zeros_like(delta_hess)
                    grad = torch.zeros_like(delta_grad)
                hess += delta_hess
                grad += delta_grad
                del delta_hess, delta_grad
                torch.cuda.empty_cache()
            
        # Calculate and apply simplified cubic Newton step
        eigenvalues_hess, eigenvectors_hess = torch.linalg.eigh(hess)
        hess_cond = torch.linalg.cond(hess).item()
        x = SICOracle.get_flat_params(name_list=weight_names)
        lipschitz_constant = 1.e-2
        eigenvectors_gradient_product = eigenvectors_hess.T.conj() @ grad
        while True:
            delta, direction = find_delta(eigenvectors_hess, eigenvalues_hess, lipschitz_constant,
                                           eigenvectors_gradient_product)
            curr_params = x + mu * direction
            SICOracle.set_flat_params(curr_params, name_list=weight_names)
            # Calculate loss value on whole signal length
            with torch.no_grad():
                tmp_loss_val = accum_loss(train_dataset)
            lin_part = 2 * (grad * direction.conj()).sum()
            quad_part = ((hess @ direction) * direction.conj()).sum()
            # Check whether model parameters are complex or not
            if not torch.is_complex(direction):
                lin_part /= 2
            cube_estimation = loss_val_train + lin_part + quad_part + lipschitz_constant * delta ** 2 * (1 + delta / 6)
            if tmp_loss_val.real > cube_estimation.real or tmp_loss_val.real > loss_val_train.real:
                lipschitz_constant *= 1.5
            else:
                break
        
        # Track algorithm parameters
        reg_param_curve.append(lipschitz_constant * delta / 4)
        loss_val_train = tmp_loss_val
        grad_norm = torch.norm(grad).item()
        grad_norm_curve.append(grad_norm)
        weights_norm_curve.append(torch.norm(curr_params).item())

        del grad, hess
        torch.cuda.empty_cache()

        # Track NMSE values on validation and test dataset and save gradient, model parameters norm and 
        # algorithm regularization history
        with torch.no_grad():
            criterion_val_train = quality_criterion(model, train_dataset)
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)

            learning_curve_test.append(loss_val_test)
            learning_curve_train.append(loss_val_train)
            learning_curve_validate.append(loss_val_validate)
            learning_curve_test_qcrit.append(criterion_val_test)
            learning_curve_train_qcrit.append(criterion_val_train)
            learning_curve_validate_qcrit.append(criterion_val_validate)

            if criterion_val_test < best_criterion_test:
                best_criterion_test = criterion_val_test
                torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
            if epoch % save_every == 0:
                np.save(save_path + f'lc_train{exp_name}.npy', np.array(learning_curve_train))
                np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
                np.save(save_path + f'lc_validate{exp_name}.npy', np.array(learning_curve_validate))
                np.save(save_path + f'lc_qcrit_train{exp_name}.npy', np.array(learning_curve_train_qcrit))
                np.save(save_path + f'lc_qcrit_test{exp_name}.npy', np.array(learning_curve_test_qcrit))
                np.save(save_path + f'lc_qcrit_validate{exp_name}.npy', np.array(learning_curve_validate_qcrit))
                np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
                np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
                np.save(save_path + f'regular{exp_name}.npy', np.array(reg_param_curve))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, " + \
                f"loss_train = {loss_val_train:.8f}, " + \
                f"quality_criterion_train = {criterion_val_train:.8f}, stepsize = {mu:.6e}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}, " + \
                f"regular param = {reg_param_curve[-1]:.4e}, " + \
                f"Hessian conditioning: {hess_cond:.4e}")
        epoch += 1

        general_timer.__exit__()
        print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test
