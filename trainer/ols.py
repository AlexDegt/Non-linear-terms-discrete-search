import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import itertools
from itertools import product

from .ls import train_ls

import sys, os
sys.path.append('../../')

from utils import Timer
from oracle import Oracle

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
OptionalList = Union[List[int], None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_ols(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                                   test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, config: dict,
                                   batch_to_tensors: BatchTensorType, tensors_to_batch: BatchTensorType, chunk_num: OptionalInt = None, 
                                   save_path: OptionalStr = None, exp_name: OptionalStr = None, weight_names: StrOrList = None,
                                   delays_range: OptionalList = None, iter_num: OptionalInt = None):
    """
    Function implements Orthogonal Least Squares (OLS) algorithm for delays search. 

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
        config (dict): Content of config file in a view if dictionary.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
        tensors_to_batch (Callable): Function, which takes batch, input and desired tensors and puts tensors into batch.
            Used to update OLS error.
        chunk_num (int, optional): The number of chunks in dataset. Defaults to "None".
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".
        delays_range (list of int, optional): Specifies list of possible delays to be chosen for the input signal. 
            If None delays_range is set to [-15, 15, 3]. Defaults to None.
        iter_num (int, optional): Number of OLS iterations. Each new iteration corresponds to 1 new model branch.
            If None, then it is set to 1. Defaults to None.

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """

    general_timer = Timer()
    general_timer.__enter__()

    # Initialize Mixed-Newton oracle
    SICOracle = Oracle(model, loss_fn)

    if delays_range is None:
        delays_range = [-15, 15, 3]

    comb_delays = list(product(*[set(np.arange(*delays_range)) for _ in range(3)]))
    # comb_delays = list(itertools.combinations_with_replacement(np.arange(*delays_range), 3))
    
    # Save all delays combinations
    np.save(os.path.join(save_path, "comb_delays.npy"), np.array(comb_delays))
    # print(len(comb_delays))
    # sys.exit()

    best_inds, best_delays = [], []
    # Initial terms matrix
    terms = []
    for batch in train_dataset:
        terms.append(torch.tensor([], device=model.device, dtype=model.dtype))
    # Iterations of OLS algorithm
    for j_iter in range(iter_num):

        # Apply LS for each delays combination and find the best one    
        timer_ls = Timer()
        timer_ls.__enter__()
        
        print(f"Iter {j_iter}. Start to calculating LS for all {len(comb_delays)} delays combinations:")
        delays_perform = []
        for j_delay_set, delay_set in enumerate(comb_delays):
            timer_single_ls = Timer()
            timer_single_ls.__enter__()
            model.set_delays([list(delay_set)])
            _, best_criterion = train_ls(model, train_dataset, train_dataset, train_dataset, loss_fn, 
                                        quality_criterion, config, batch_to_tensors, chunk_num, 
                                        save_path, exp_name, weight_names)
            delays_perform.append(best_criterion)
            # np.save(os.path.join(save_path, f"delays_perform_iter_{j_iter + 1}.npy"), np.array(delays_perform))
            timer_single_ls.__exit__()
            print(f"Calculated {j_delay_set + 1} of {len(comb_delays)} delays combs. Elapsed {timer_single_ls.interval:.3f} s.")
        
        timer_ls.__exit__()
        print(f"Iter {j_iter}. Totally elapsed {timer_ls.interval:.3f} s per all {len(comb_delays)} delays combinations.")
            
        best_inds.append(np.argmin(delays_perform))
        curr_best_delays = comb_delays[best_inds[-1]]
        best_delays.append(list(curr_best_delays))
        # Save best delays combinations
        np.save(os.path.join(save_path, "best_delays.npy"), np.array(best_delays))
        
        # Set best delays to the model and calculate term matrix (jacobian)
        # Pay attention, that whole terms matrix is constructed every step once again!
        # Such approach alows to save memory by division whole signal into batches.
        SICOracle._model.set_delays(best_delays)

        # Calculate projector in 2 steps (loops)
        proj_timer = Timer()
        proj_timer.__enter__()
        print(f"Iter {j_iter}. Begin to implement OLS error projection step:")
        # Step 1:
        for j, batch in enumerate(train_dataset):
            jac = SICOracle.jacobian(batch, batch_to_tensors)
            jac_H = torch.conj(torch.permute(jac, (0, 2, 1)))
            residual = batch_to_tensors(batch)[1]
            if j == 0:
                hess = torch.bmm(jac_H, jac)
                grad = torch.bmm(jac_H, torch.permute(residual, (0, 2, 1)))
            else:
                hess += torch.bmm(jac_H, jac)
                grad += torch.bmm(jac_H, torch.permute(residual, (0, 2, 1)))
        hess_inv = []
        for i in range(hess.size()[0]):
            hess_inv.append(torch.linalg.pinv(hess[i, ...], rcond=1e-15, hermitian=True)[None, ...])
        hess_inv = torch.cat(hess_inv, dim=0)
        direction = torch.bmm(hess_inv, grad)
        # Step 2:
        for j, batch in enumerate(train_dataset):
            jac = SICOracle.jacobian(batch, batch_to_tensors)
            delta_residual = torch.bmm(jac, direction)
            residual = batch_to_tensors(batch)[1]
            inp = batch_to_tensors(batch)[0]
            # Update residual vector (desired) in training dataset
            # if j_iter == 0:
            #     np.save(os.path.join(save_path, f"residual_{0}.npy"), residual.detach().cpu().numpy()[0, 0, :])
            residual -= torch.permute(delta_residual, (0, 2, 1))
            tensors_to_batch(batch, inp, residual)
            # np.save(os.path.join(save_path, f"jac_{j_iter + 1}.npy"), jac[0, ...].detach().cpu().numpy())
            # np.save(os.path.join(save_path, f"delta_residual_{j_iter + 1}.npy"), torch.permute(delta_residual, (0, 2, 1)).detach().cpu().numpy()[0, 0, :])
            # np.save(os.path.join(save_path, f"residual_{j_iter + 1}.npy"), residual.detach().cpu().numpy()[0, 0, :])
        proj_timer.__exit__()
        print(f"Iter {j_iter}. Projection finished, time elapsed {proj_timer.interval} s")
        
    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return best_delays