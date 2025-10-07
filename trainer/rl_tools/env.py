import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch.nn as nn

from contextlib import contextmanager
from collections import defaultdict

from typing import List, Tuple, Union, Callable, Iterable

from copy import deepcopy, copy
import sys

import pandas as pd

RangeType = List[int]
OptionalList = Union[None, List]

@contextmanager
def no_print():
    import sys, io
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout

class IntRangeSpace(gym.Space):
    """
        Auxiliary object to create sample delays in range [low, high]
    """
    def __init__(self, low, high):
        super().__init__(shape=(), dtype=np.int32)
        self.low = low
        self.high = high

    def sample(self):
        return np.random.randint(self.low, self.high + 1)

    def contains(self, x):
        return self.low <= x <= self.high

class PerformanceEnv(gym.Env):
    """
        Current environment is designed to check RL algorithm performance
        for signal processing model hyperparameter optimization task.
        
        Environment state is model input delays.
        Agent step is a change input delays. It includes indices of delays
            to perform step, step value [-step_max, step_max]. The number of 
            delays to be changed at current step is regulated by parameter delays2change_num.
        Reward for agent steps is represented by MSE (Mean Square Error) of model for chosen delays set.

        Agent provides strategy for current environment, recalculates delays and finds MSE reward.

        Args:
            tomb_raider (torch.nn.Module): The signal processing model, which hyperparameters are implied to be optimized.
            delays_number (int): Required number of delays in model.
            delays_range (list of ints): List, whice inludes minimum and maximum delays possible to set.
            max_delay_step (int): Maximum value of delay step, which can be performed by an agent.
            delays2change_num (int): Number of delays, which required to be changed per each step.
            max_steps (int): Maximum number of steps in each episode.
            train_tomb_raider (partial function): Pointer to the function with fixed parameters.
            start_mode (str): Flag, which shows how to start new trajectory:
                start_mode == 'zeros', - zero delays,
                start_mode == 'random', - delays from uniform distribution,
                start_mode == 'continue' - continues from saved state. First trajectory - from zeros.
                start_mode == 'same', - start each trajectory of init_delays.
            init_delays (list, optional): Initial delays to start from each trajectory if start_mode=='same'.
    """
    def __init__(self, tomb_raider: nn.Module, delays_number: int, delays_range: RangeType, max_delay_step: int, delays2change_num: int, max_steps: int,
                 train_tomb_raider, start_mode: str='zeros', init_delays: OptionalList=None):
        super(PerformanceEnv, self).__init__()

        assert len(delays_range) == 2 and all([type(delay) == int for delay in delays_range]), "Delays range must include 2 integers."
        assert 0 < delays2change_num <= delays_number, "Number of delays to be changed per each step must higher than 0 and not higher than number of delays."
        assert delays_range[0] < delays_range[1]

        # Save environment parameters
        self.delays_number = delays_number
        self.__delays_range = delays_range
        self.__max_delay_step = max_delay_step
        self.delays2change_num = delays2change_num
        self.__train_tomb_raider = train_tomb_raider
        self.tomb_raider = tomb_raider
        # Number of delays in each branch. For 2D model it equals 3
        self.__delays_in_branch = len(self.tomb_raider.delays[0])

        # Definition of state (observation) space
        self.state_space = spaces.Box(low=delays_range[0], high=delays_range[1] - 1, shape=(delays_number,), dtype=int)

        # Definition of action spaces
        single_pair_space = spaces.Tuple((
            spaces.Discrete(delays_number),       # Index of delay to be changed
            spaces.Discrete(2 * max_delay_step + 1)
            # IntRangeSpace(-max_delay_step, max_delay_step)      # delays step -max_delay_step..max_delay_step
        ))
        self.action_space = spaces.Tuple([single_pair_space for _ in range(delays2change_num)])
        
        if start_mode == 'same':
            assert init_delays is not None, "Please, define init_delays for start_mode=\'same\'."
            assert type(init_delays) == list, f"init_delays must of a list type, but {type(init_delays)} is given."

        self.init_delays = init_delays
        self.start_mode = start_mode

        # Internal state
        if init_delays is not None:
            self.state = sum(init_delays, [])
            assert len(self.state) == delays_number, f"Number of delays in init_delays must be the same branch_number * delays_per_branch."
            self.init_delays = deepcopy(self.state)
        else:
            self.state = np.zeros(delays_number, dtype=int).tolist()
        self.step_curr = 1 # Current step number
        self.max_steps = max_steps

        # Store best performance value (dB) and corresponding delays
        self.best_perform = 100
        self.best_delays = []

        # Dataframe to save delays sets already researched by Oracle
        self.oracle_buffer = pd.DataFrame(columns=['state', 'reward'])
        self.oracle_call_num = 0

    def reset(self, seed=None):
        """
            Resets environment
            
            Args:
                seed (int): random seed.                    
        """
        super().reset(seed=seed)
        self.step_curr = 1
        if self.start_mode == 'zeros':
            self.state = np.zeros((self.delays_number,), dtype=int).tolist()
        elif self.start_mode == 'random':
            self.state = self.state_space.sample()
        elif self.start_mode == 'continue':
            pass
        elif self.start_mode == 'same':
            # print('hey')
            self.state = deepcopy(self.init_delays)
            # print(f"Just after reset: {self.state}")
        return self.state, {}

    def step(self, action):
        """
            Makes agent step in the environment.
            For each pair in action tuple chenges delays (state).
            Updated delays are put into the optimized model. Then reward (MSE) is calculated.

            Returns:
                Tuple, which includes new state, reward for the step, flags terminated, truncated and um info, if needed.
                    truncated is True, is maximum number of steps is achieved,
                    terminated is always False, since agent can walk in the environment without limits.
        """
        assert self.action_space.contains(action), f"Chosen action is out of action space."

        terminated = False
        truncated = False
        if self.step_curr >= self.max_steps:
            truncated = True
        # Increase step
        self.step_curr += 1

        for action_pair in action:
            delay_ind, delay_step_ind = action_pair
            # steps = np.arange(-self.__max_delay_step, self.__max_delay_step + 1, 1)
            # steps = steps[steps != 0]
            # delay_step = steps[delay_step_ind]
            delay_step = self.step_ind_to_step(delay_step_ind)
            # print(self.state)
            # Bounce from the bound
            # L = self.state_space.low[0]
            # U = self.state_space.high[0]
            # while self.state[delay_ind] + delay_step < L or self.state[delay_ind] + delay_step > U:
            #     if self.state[delay_ind] + delay_step < L:
            #         delay_step = 2 * L - (2 * self.state[delay_ind] + delay_step)
            #     elif self.state[delay_ind] + delay_step > U:
            #         delay_step = 2 * U - (2 * self.state[delay_ind] + delay_step)
            self.state[delay_ind] += delay_step
            # Clip state if its out of range after step. Clip is not good for training
            self.state = np.clip(self.state, self.state_space.low, self.state_space.high).tolist()
            # print(self.state, action)
            # sys.exit()
        assert self.state_space.contains(np.asarray(self.state)), f"Chosen state is out of state space."
        # Search reward inside the oracle buffer.
        # If ther if no such state in buffer, call oracle
        if self.state in self.oracle_buffer['state'].tolist():
            reward = self.oracle_buffer.loc[self.oracle_buffer["state"].apply(lambda x: x == self.state), "reward"].iloc[0]
        else:        
            # Reshape delays according to their position in branches
            delays = np.array(self.state).reshape(-1, self.__delays_in_branch).tolist()
            # Load delays in model
            self.tomb_raider.set_delays(delays)
            # Train tomb raider in a search of high performance!
            with no_print():
                _, perform_db = self.__train_tomb_raider()
            # Reward design...
            # reward = (10 ** (-1 * perform_db / 10) / 40 - 0.5) / 100
            reward = -1 * perform_db
            self.oracle_buffer = pd.concat([
                self.oracle_buffer,
                pd.DataFrame([{'state': copy(self.state), 'reward': reward}])
            ], ignore_index=True)
            # Store best performance and delays
            if perform_db < self.best_perform:
                self.best_perform = perform_db
                self.best_delays = delays
        # print(self.state)

        return self.state, reward, terminated, truncated, {}

    def step_ind_to_step(self, delay_step_ind):
        steps = np.arange(-self.__max_delay_step, self.__max_delay_step + 1, 1)
        steps = steps[steps != 0]
        if not isinstance(delay_step_ind, np.ma.MaskedArray):
            return steps[delay_step_ind]
        else:
            return delay_step_ind

    def render(self):
        print("State:", self.state)

    def close(self):
        pass

class NormalizeWrapper(gym.Wrapper):
    """
        Object for states and rewards normaliation by running mean and std:
        mean_t = mean_{t-1} + alpha * (o_t - mean_{t-1})
        std_t^2 = std_{t-1}^2 + alpha * ((o_t - mean_{t-1}) - std_{t-1}^2)

        Better to set alpha ~ 0.01 when environment is static,
        better to set alpha 0.5-0.9 when environment is highly dynamic.
    """
    def __init__(self, env, state_alpha=0.01, reward_alpha=0.01, epsilon=1e-8):
        super().__init__(env)
        
        # Normalization for state
        self.state_mean = np.zeros(env.state_space.shape, dtype=np.float32)
        self.state_var = np.ones(env.state_space.shape, dtype=np.float32)
        self.state_alpha = state_alpha
        self.epsilon = epsilon
        
        # Normalization for reward
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_alpha = reward_alpha

    def _normalize_state(self, state):
        # Recalculate statement w.r.t. running normalization
        delta = state - self.state_mean
        self.state_mean += self.state_alpha * delta
        self.state_var += self.state_alpha * (delta**2 - self.state_var)
        
        normalized_state = (state - self.state_mean) / np.sqrt(self.state_var + self.epsilon)
        return normalized_state.tolist()

    def _normalize_reward(self, reward):
        # Recalculate reward w.r.t. running normalization
        delta = reward - self.reward_mean
        self.reward_mean += self.reward_alpha * delta
        self.reward_var += self.reward_alpha * (delta**2 - self.reward_var)
        
        normalized_reward = (reward - self.reward_mean) / np.sqrt(self.reward_var + self.epsilon)
        return normalized_reward
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return state, info
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_state(state), self._normalize_reward(reward), terminated, truncated, info

    def get_normalized_state(self):
        return self._normalize_state(self.env.state)

    def get_normalized_reward(self, reward):
        return self._normalize_reward(reward)

    def __getattr__(self, name):
        return getattr(self.env, name)

class TrajectoryNormalizeWrapper(gym.Wrapper):
    """
        Object for normaization of states and rewards for the
        whole trajectory. Since distribution of rewards in environment
        is static, then better to apply normalization at the end of each episode.
    """
    def __init__(self, env, epsilon=1.e-8):
        super().__init__(env)
        self.states = []
        self.rewards = []
        self.epsilon = epsilon

    def reset(self, **kwargs):
        self.states = []
        self.rewards = []
        state, info = self.env.reset(**kwargs)
        # self.states.append(state)
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.states.append(state)
        self.rewards.append(reward)
        return state, reward, terminated, truncated, info

    def normalize_trajectory(self):
        states = np.array(self.states, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        states_mean = states.mean()
        states_std = states.std() + self.epsilon
        rewards_mean = rewards.mean()
        rewards_std = rewards.std() + self.epsilon

        norm_states = (states - states_mean) / states_std
        norm_rewards = (rewards - rewards_mean) / rewards_std
        return norm_states, norm_rewards

    def __getattr__(self, name):
        return getattr(self.env, name)

class EnvRunner:
    """ Reinforcement learning runner in an environment with given policy """

    def __init__(self, env, policy, nsteps, transforms=None, step_var=None):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.state = {"latest_observation": self.env.reset()[0]}

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    def reset(self, **kwargs):
        """ Resets env and runner states. """
        self.state["latest_observation"], info = self.env.reset(**kwargs)
        tmp = self.state["latest_observation"]
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        time_steps = []
        self.state["env_steps"] = self.nsteps

        for i in range(self.nsteps):
            observations.append(deepcopy(self.state["latest_observation"]))
            time_steps.append(i)
            norm_param = max(max(abs(self.env.state_space.high)), max(abs(self.env.state_space.low)))
            inputs = {"state": observations[-1] / norm_param,
                      "time": [time_steps[-1]]}
            act = self.policy.act(inputs)
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)

            obs, rew, terminated, truncated, _ = self.env.step(trajectory['actions'][-1])

            done = np.logical_or(terminated, truncated)
            self.state["latest_observation"] = deepcopy(obs)
            rewards.append(rew)
            resets.append(done)

        trajectory.update(
            observations=observations,
            rewards=rewards,
            resets=resets,
            time_steps=time_steps)
        trajectory["state"] = deepcopy(self.state)

        for transform in self.transforms:
            transform(trajectory)
        
        return trajectory

class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None
        self.trajectory_count = 0

    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        trajectory_len = self.trajectory["observations"].shape[0]

        permutation = np.random.permutation(trajectory_len)
        for key, value in self.trajectory.items():
            if key != 'state':
                self.trajectory[key] = value[permutation, ...]

    def get_next(self, return_whole=False):
        """ Returns next minibatch.  """
        if not self.trajectory:
            self.trajectory = self.runner.get_next()
            for transform in self.transforms:
                transform(self.trajectory)
            self.whole_trajectory = deepcopy(self.trajectory)
            # np.save(f"observations_{self.trajectory_count}.npy", np.asarray(self.trajectory["observations"]))

        if self.minibatch_count == self.num_minibatches:
            # self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count += 1

        if self.epoch_count == self.num_epochs:
            self.runner.reset()
            self.trajectory = self.runner.get_next()
            self.trajectory_count += 1
            for transform in self.transforms:
                transform(self.trajectory)
            self.whole_trajectory = deepcopy(self.trajectory)
            # np.save(f"observations_{self.trajectory_count}.npy", np.asarray(self.trajectory["observations"]))

            # self.shuffle_trajectory()
            self.minibatch_count = 0
            self.epoch_count = 0

        trajectory_len = self.trajectory["observations"].shape[0]

        batch_size = trajectory_len//self.num_minibatches

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != 'state':
                minibatch[key] = value[self.minibatch_count*batch_size: (self.minibatch_count + 1)*batch_size]

        self.minibatch_count += 1

        # for transform in self.transforms:
        #     transform(minibatch)
        if return_whole:
            return minibatch, self.whole_trajectory
        else:
            return minibatch

class TrajectorySampler_v1_1:
    """ 
        Samples minibatches from trajectory for a number of epochs.
        Minibatches firslty formed as tensors of shape [N, B],
        where N - number of trajectories in batch, B - length of batch of trajectory.
    """
    def __init__(self, runner, num_epochs, num_minibatches, traj_per_batch, transforms=None, mask_max=False):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.traj_per_batch = traj_per_batch
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None
        self.trajectory_count = 0
        self.mask_max = mask_max

    def sample_trajs(self):
        for j_traj in range(self.traj_per_batch):
            self.runner.reset()
            trajectory = self.runner.get_next()
            if j_traj == 0:
                trajectory_batch = deepcopy(trajectory)
            else:
                for key, val in trajectory_batch.items():
                    if key != 'state' and key != 'latest_observation' and key != 'env_steps':
                        if j_traj == 1:
                            trajectory_batch[key] = trajectory_batch[key][None, :]
                        trajectory_batch[key] = np.concatenate((trajectory_batch[key], trajectory[key][None, :]), axis=0)
        return trajectory_batch

    def get_next(self, return_whole=False):
        """ Returns next minibatch.  """
        if not self.trajectory or self.minibatch_count == self.num_minibatches:
            self.minibatch_count = 0

            self.trajectory = self.sample_trajs()

            trajectory_len = self.trajectory["observations"].shape[1]

            batch_size = trajectory_len // self.num_minibatches

            # Mask all elements of trajectory after maximum return per each trajectory
            if self.mask_max:
                cols = np.arange(self.trajectory['returns'].shape[1])
                mask = cols[None, :] > np.argmax(self.trajectory['rewards'], axis=1)[:, None]
                mask_action = np.concatenate((mask[..., None, None], mask[..., None, None]), axis=-1)
                mask_state = np.concatenate((mask[..., None], mask[..., None], mask[..., None]), axis=-1)
                for key, val in self.trajectory.items():
                    if key != 'state':
                        if key == "actions" or key == "log_probs":
                            self.trajectory[key] = np.ma.array(self.trajectory[key], mask=mask_action, dtype=self.trajectory[key].dtype)
                        elif key == "observations":
                            self.trajectory[key] = np.ma.array(self.trajectory[key], mask=mask_state, dtype=int)
                        else:
                            self.trajectory[key] = np.ma.array(self.trajectory[key], mask=mask, dtype=self.trajectory[key].dtype)
                self.trajectory['mask'] = mask
            else:
                self.trajectory['mask'] = np.zeros_like(self.trajectory['rewards'], dtype=int)


            # Normalize returns in whole trajectory
            for transform in self.transforms:
                transform(self.trajectory)         

        minibatch = {}
        for key, value in self.trajectory.items():
            if key != 'state':
                minibatch[key] = value[:, self.minibatch_count*batch_size: (self.minibatch_count + 1)*batch_size, ...].reshape(-1, *value.shape[2:])

        for key, value in self.trajectory.items():
            if key != 'state':
                self.trajectory[key] = self.trajectory[key].reshape(-1, *self.trajectory[key].shape[2:])

        self.minibatch_count += 1

        if return_whole:
            return minibatch, self.trajectory
        else:
            return minibatch