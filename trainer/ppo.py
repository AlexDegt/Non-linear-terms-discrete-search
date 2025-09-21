import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import itertools
from functools import partial
from collections import defaultdict

from .ls import train_ls

import sys, os
sys.path.append('../../')

from utils import Timer
from oracle import Oracle
from .rl_tools import PerformanceEnv, NormalizeWrapper, TrajectoryNormalizeWrapper, EnvRunner
from .rl_tools import CNNSharedBackPolicy, MLPSharedBackPolicy, Policy

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
OptionalList = Union[List[int], None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_ppo(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                                   test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, config: dict,
                                   batch_to_tensors: BatchTensorType, tensors_to_batch: BatchTensorType, chunk_num: OptionalInt = None, 
                                   save_path: OptionalStr = None, exp_name: OptionalStr = None, weight_names: StrOrList = None,
                                   delays_range: OptionalList = None, iter_num: OptionalInt = None):
    """
    Function implements Proximal Policy Optimization algorithm for optimal delays search.

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
    # Function to calculate MSE reward
    train_tomb_raider = partial(train_ls, model, train_dataset, train_dataset, train_dataset, loss_fn, 
                                        quality_criterion, config, batch_to_tensors, chunk_num, 
                                        save_path, exp_name, weight_names)

    env = PerformanceEnv(model, delays_number, delays_range, max_delay_step, delays2change_num, max_steps, train_tomb_raider)

    # Normalization state and reward parameters
    state_alpha = config["state_alpha"]
    reward_alpha = config["reward_alpha"]

    # env = TrajectoryNormalizeWrapper(env)
    env = NormalizeWrapper(env)
    # Better to make normalization at the end of episode
    # state, info = env.reset(seed=0, zero_start=True)

    # states, rewards = [], []
    # for _ in range(60):
    #     action = env.action_space.sample()
    #     state, reward, terminated, truncated, info = env.step(action)
    #     states.append(state)
        # rewards.append(reward)
        # print(state, reward)
    # states = np.array(states)
    # rewards = np.array(rewards)
    # print(states)
    # print(rewards)
    # sys.exit()
    # states, rewards = env.normalize_trajectory()

    # action = env.action_space.sample()
    # print(type(action[0][0]), type(action[0][1]))
    # print(action)
    # print(np.array(action).shape)
    # state = env.state_space.sample()

    state_dim = len(model.delays[0])
    delays_steps_num = 2 * max_delay_step + 1
    hidden_shared_size=4
    hidden_shared_num=2
    kernel_shared_size=3
    hidden_separ_size=4
    hidden_separ_num=2
    kernel_separ_size=3

    # agent = CNNSharedBackPolicy(state_dim, delays2change_num, delays_steps_num,
    #                             hidden_shared_size, hidden_shared_num, kernel_shared_size,
    #                             hidden_separ_size, hidden_separ_num, kernel_separ_size,
    #                             model.device)
    agent = MLPSharedBackPolicy(state_dim, delays2change_num, delays_steps_num,
                                hidden_shared_size, hidden_shared_num,
                                hidden_separ_size, hidden_separ_num,
                                model.device)
    agent.count_parameters()

    # policy, value = agent(states_batched)

    policy = Policy(agent)

    # policy.act(states)
    # sys.exit()

    runner = EnvRunner(env, policy, 30)
    trajectory = runner.get_next()

    print(trajectory)

    # {k: v.shape for k, v in trajectory.items() if k != "state"}


    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return best_delays