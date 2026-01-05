import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import itertools
from functools import partial
from collections import defaultdict

from .ls import train_ls
import time
from copy import deepcopy

import sys, os
sys.path.append('../../')

from utils import Timer
from oracle import Oracle
from .rl_tools import PerformanceEnv, NormalizeWrapper, TrajectoryNormalizeWrapper, EnvRunner, EnvRunnerMemory, TrajectorySampler_v1_1, TrajectorySampler, TrajectorySamplerMemory
from .rl_tools import MLPSepDelayStep, MLPSepDelaySepStep, MLPSepDelaySepStepStepID, MLPConditionalStep, LSTMShared, PolicyActor, Policy_v1_3, PolicyMemory
from .rl_tools import AccumReturn, AsArray, NormalizeReturns, TrainingTracker
from .rl_tools import PolicyGradient

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
OptionalList = Union[List[int], None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_pg(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                                   test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, config: dict,
                                   batch_to_tensors: BatchTensorType, tensors_to_batch: BatchTensorType, chunk_num: OptionalInt = None, 
                                   save_path: OptionalStr = None, exp_name: OptionalStr = None, weight_names: StrOrList = None,
                                   delays_range: OptionalList = None, iter_num: OptionalInt = None):
    """
    Function implements simple Policy Gradient algorithm for optimal delays search.

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

    if delays_range is None:
        delays_range = [-15, 15, 1]

    comb_delays = list(itertools.combinations_with_replacement(np.arange(*delays_range), 3))
    
    # Save all delays combinations
    np.save(os.path.join(save_path, "comb_delays.npy"), np.array(comb_delays))

    best_inds, best_delays = [], []
    
    # Environment parameters
    delays_number = sum([len(delay_set) for delay_set in model.delays])
    delays_range = config["delays_range"]
    max_delay_step = config["max_delay_step"]
    delays2change_num = config["delays2change_num"]
    max_steps = config["max_steps"]
    start_mode = config["start_mode"]
    init_delays = config["init_delays"]
    # Function to calculate MSE reward
    train_tomb_raider = partial(train_ls, model, train_dataset, train_dataset, train_dataset, loss_fn, 
                                        quality_criterion, config, batch_to_tensors, chunk_num, 
                                        save_path, exp_name, weight_names)

    # Define environment for best MSE search
    env = PerformanceEnv(model, delays_number, delays_range, max_delay_step, delays2change_num, max_steps, train_tomb_raider, start_mode, init_delays)

    # Define normalization wrapper for environment
    state_alpha = config["state_alpha"]
    reward_alpha = config["reward_alpha"]
    env = NormalizeWrapper(env, state_alpha, reward_alpha)

    # General training parameters
    num_runner_steps = config["num_runner_steps"]
    gamma = config["gamma"]
    accum_return_mode = config["accum_return_mode"]
    num_epochs_per_traj = 1
    num_minibatches = config["num_minibatches"]
    traj_per_batch = config["traj_per_batch"]
    mask_max = config["mask_max"]

    # Define parameters of agent
    state_dim = len(model.delays[0]) * len(model.delays)
    delays_steps_num = 2 * max_delay_step# + 1
    # hidden_shared_size = config["hidden_shared_size"]
    # hidden_shared_num = config["hidden_shared_num"]
    hidden_delay_ind_size = config["hidden_delay_ind_size"]
    hidden_delay_ind_num = config["hidden_delay_ind_num"]
    hidden_delay_step_size = config["hidden_delay_step_size"]
    hidden_delay_step_num = config["hidden_delay_step_num"]
    stepid_embed_size = config["stepid_embed_size"]
    ind_choice_embed_size = config["ind_choice_embed_size"]
    hidden_shared_size = config["hidden_shared_size"]
    hidden_shared_num = config["hidden_shared_num"]
    hidden_size = config["hidden_size"]
    # agent = MLPConditionalStep(state_dim, delays2change_num, delays_steps_num,
    #                         num_runner_steps, stepid_embed_size, ind_choice_embed_size,
    #                         hidden_shared_size, hidden_shared_num,
    #                         hidden_delay_ind_size, hidden_delay_ind_num,
    #                         hidden_delay_step_size, hidden_delay_step_num,
    #                         model.device)
    # agent = MLPSepDelaySepStepStepID(state_dim, delays2change_num, delays_steps_num,
    #                         num_runner_steps, stepid_embed_size,
    #                         hidden_delay_ind_size, hidden_delay_ind_num,
    #                         hidden_delay_step_size, hidden_delay_step_num,
    #                         model.device)
    agent = LSTMShared(state_dim, delays2change_num, delays_steps_num,
                       num_runner_steps, stepid_embed_size, hidden_size, 
                       model.device)
    agent.count_parameters()
    # agent.enumerate_parameters()

    # Policy: different returns for trajectory sampling and agent training
    # policy = PolicyActor(agent)
    # policy = Policy_v1_3(agent)
    policy = PolicyMemory(agent)

    def make_ppo_runner(env, policy, num_runner_steps=2048, gamma=0.99, 
                        num_epochs=10, num_minibatches=32):
        """ Creates runner for PPO algorithm. """
        runner_transforms = [AsArray(), AccumReturn(policy, gamma=gamma, mode=accum_return_mode)]
        # runner = EnvRunnerMemory(env, policy, num_runner_steps, transforms=runner_transforms)
        # Use default EnvRunner for memory agent!
        runner = EnvRunner(env, policy, num_runner_steps, transforms=runner_transforms)

        sampler_transforms = [NormalizeReturns()]
        # sampler_transforms = []
        # sampler = TrajectorySampler(runner, num_epochs=num_epochs, 
        #                 num_minibatches=num_minibatches,
        #                 traj_per_batch=traj_per_batch,
        #                 transforms=sampler_transforms,
        #                 mask_max=mask_max)
        # sampler = TrajectorySampler_v1_1(runner, num_epochs=num_epochs, 
        #                         num_minibatches=num_minibatches,
        #                         traj_per_batch=traj_per_batch,
        #                         transforms=sampler_transforms,
        #                         mask_max=mask_max)
        sampler = TrajectorySamplerMemory(runner, num_epochs=num_epochs, 
                                num_minibatches=num_minibatches,
                                traj_per_batch=traj_per_batch,
                                transforms=sampler_transforms,
                                mask_max=mask_max)
        return sampler

    runner = make_ppo_runner(env, policy, num_runner_steps, gamma, num_epochs_per_traj, num_minibatches)
    
    # Optimizer parameters
    lr = config["lr"]
    eps = config["eps"]
    optimizer = torch.optim.Adam(policy.agent.parameters(), lr=lr, eps=eps)
    epochs = config["total_epoch_num"]
    # Learning rate scheduler
    # lr_mult = lambda epoch: (1 - (epoch/epochs))
    # lr_mult = lambda epoch: 1 - (1 - 1e-3) * (epoch / epochs)
    lr_mult = lambda epoch: 1
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)

    explore_loss_coef = config["explore_loss_coef"]
    max_grad_norm = config["max_grad_norm"]
    # Define Policy Gradient RL optimizer
    pg = PolicyGradient(policy, optimizer, explore_loss_coef, max_grad_norm)

    # Define statistics tracker
    alg_type = 'pg'
    log_every_epochs = config["log_every_epochs"]
    log_trajs = config["log_trajs"]
    tracker = TrainingTracker(env, pg, traj_per_batch, log_every_epochs, log_trajs, save_path, alg_type)
    
    for epoch in range(epochs):
        t_epoch_start = time.time()

        minibatch, trajectory = runner.get_next(return_whole=True)

        tracker.save_oracle_buffer()

        pg.step(minibatch)

        sched.step()
        tracker.accum_stat(trajectory)

        # Calls one more act through minibatch trajectories
        tracker.approx_kl(trajectory)
        tracker.log_steps(trajectory, epoch)

        t_epoch_end = time.time()
        print(f"Epoch {epoch}, reward max mean = {tracker.rewards_max_mean[-1]:.5f}, time per epoch {(t_epoch_end - t_epoch_start):.3f} s")

    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return best_delays