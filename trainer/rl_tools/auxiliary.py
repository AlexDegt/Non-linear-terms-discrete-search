import numpy as np
import torch
import os, sys
import pandas as pd
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", message=".*converting a masked element to nan.*")

class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        
    def __call__(self, trajectory):
        gamma = self.gamma
        lam = self.lambda_

        rewards = np.asarray(trajectory["rewards"], dtype=np.float32).flatten()
        values  = np.asarray(trajectory["values"],  dtype=np.float32).flatten()
        dones   = np.asarray(trajectory["resets"],  dtype=np.float32).flatten()

        T = len(rewards)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0 
                mask = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                mask = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae

        value_targets = advantages + values

        trajectory["advantages"] = advantages
        trajectory["value_targets"] = value_targets
        return trajectory

class AccumReturn:
    """ 
        Calculate accumulative return.
        If mode == 'terminal', - function copies last rewards to call returns,
            which means that only the last reward is important.
        If mode == 'discount', - __call__ calucalates discount return with
            discount parameter gamma.
        If mode == 'max', - __call__ copies maximum reward during trajectory to all steps.
    """
    def __init__(self, policy, gamma=0.99, mode='terminal'):
        self.policy = policy
        self.gamma = gamma
        assert mode == 'terminal' or mode == 'discount' or mode == 'max' or mode == 'discount_increment', \
            f"Return accumulation mode must be either \'discount\', \'terminal\', \'max\' or \'discount_increment\',, but {mode} is given."
        self.mode = mode
        
    def __call__(self, trajectory):
        gamma = self.gamma

        rewards = np.asarray(trajectory["rewards"], dtype=np.float32).flatten()
        rewards_inc = np.asarray(trajectory["rewards_inc"], dtype=np.float32).flatten()
        dones   = np.asarray(trajectory["resets"],  dtype=np.float32).flatten()
        T = len(rewards)
        returns = np.zeros_like(rewards, dtype=np.float32)

        R = 0.0
        for t in range(len(rewards)-1, -1, -1):
            if self.mode == 'discount':
                R = rewards[t] + gamma * R * (1.0 - dones[t])
                returns[t] = R
            elif self.mode == 'terminal':
                returns[t] = rewards[-1]
            elif self.mode == 'max':
                returns[t] = np.max(rewards)
            elif self.mode == 'discount_increment':
                R = rewards_inc[t] + gamma * R * (1.0 - dones[t])
                returns[t] = R

        trajectory["returns"] = returns.astype(np.float32)
        return trajectory

class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory, eps=1e-8):
        advantages = np.asarray(trajectory["advantages"]).flatten()
        var = np.var(advantages)
        mean = np.mean(advantages)
        trajectory["advantages"] = (advantages - mean) / np.sqrt(var + eps)

class NormalizeReturns:
    """ Normalizes cumulative returns to have zero mean and variance 1. """
    def __init__(self, beta=0.0):
        self.beta = beta
        self.baseline = 0
    def __call__(self, trajectory, mask=None, eps=1e-8):
        # returns = np.asarray(trajectory["returns"]).flatten()
        # returns = np.asarray(trajectory["returns"])
        returns = trajectory["returns"].copy()

        # No baseline. For max reward strategy
        var = returns[:, 0].var()
        mean = returns[:, 0].mean()
        ret = (returns - mean) / np.sqrt(var + eps)

        # No baseline. For max inrement strategy
        # var = returns.var()
        # mean = returns.mean()
        # ret = (returns - mean) / np.sqrt(var + eps)

        # var = returns.var(axis=0)
        # mean = returns.mean(axis=0)
        # ret = (returns - mean) / np.sqrt(var + eps)
        # ret = returns
        
        # Baseline. For max reward strategy
        # self.baseline = 0.99 * self.baseline + 0.01 * mean
        # adv = returns - self.baseline
        # var = adv[:, 0].var()
        # mean = adv[:, 0].mean()
        # ret = (adv - mean) / np.sqrt(var + eps)
        
        # ret = returns - 14
        # print(max(ret))
        trajectory["returns"] = ret
        # trajectory["returns"] = (returns - mean) / np.sqrt(var + eps)
        # pass

class AsArray:
    """ 
    Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
        # Modify trajectory inplace. 
        # for k, v in filter(lambda kv: kv[0] != "state" and kv[0] != "actions", trajectory.items()):
        #     trajectory[k] = np.asarray(v)
        for k, v in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[k] = np.asarray(v)

class TrainingTracker:
    """ 
        Object includes methods and attributes for agent training parameters tracking.
        Accumulates algorithm parameters during training.

        log_every_epochs (int): logs are saved every log_every_epochs epochs.
        log_trajs (list): trajectory indices range to log states and actions.
    """
    def __init__(self, env, alg, traj_per_batch, log_every_epochs: int, log_trajs: list, save_path=None, alg_type='ppo', summary_writer=None):
        
        self.env = env
        self.alg = alg
        self.alg_type = alg_type
        self.save_path = save_path
        self.traj_per_batch = traj_per_batch

        self.log_every_epochs = log_every_epochs
        self.log_trajs = log_trajs

        self.summary_writer = summary_writer
        self.summary_writer_global_step = 0

        self.log = pd.DataFrame(columns=['epoch', 'trajectory', 'state', 'action', 'reward', 'return', 'aver', 'std'])
        for j in range(self.env.delays2change_num):
            self.log[f"ind {j}"] = None
            self.log[f"step {j}"] = None

        self.log_best_traj = pd.DataFrame(columns=['epoch', 'trajectory', 'state', 'action', 'reward', 'return', 'aver', 'std'])
        for j in range(self.env.delays2change_num):
            self.log_best_traj[f"ind {j}"] = None
            self.log_best_traj[f"step {j}"] = None

        # Parameters to be tracked
        self.rewards_mean = []
        self.rewards_last_mean = []
        self.rewards_max_mean = []
        self.rewards_max = []
        self.rewards_min = []
        self.r2_score = [] # Coefficient of determination
        self.policy_entropy = []
        self.value_loss = []
        self.policy_loss = []
        self.value_targets = []
        self.value_predicts = []
        self.grad_norm = []
        self.grad_norm_policy = []
        self.grad_norm_value = []
        self.advantages = []
        self.returns = []
        self.log_policy = []
        self.best_perform_list = []
        self.best_perform = 100
        self.best_delays = []
        self.approx_kl_list = []
        self.clip_fraction_list = []

        self.min_min_index_max_reward_list = []
        self.mean_min_index_max_reward_list = []


        self.policy_ind_distr = []
        self.policy_step_distr = []

    def save_oracle_buffer(self):
        self.env.oracle_buffer.to_excel(os.path.join(self.save_path, "oracle_buffer.xlsx"), sheet_name="Oracle buffer", index=True)
    
    def log_steps(self, trajectory: dict, curr_epoch: int):
        """
            Saves excel file with whole log.

            trajectory (dict): Batched whole(!) trajectory to gather statistics from.
            curr_epoch (int): current epoch.
        """
        self.alg.policy.agent.eval()
        norm_param = max(max(abs(self.env.state_space.high)), max(abs(self.env.state_space.low)))
        with torch.no_grad():
            # Save trajectories every log_every_epochs epochs
            if curr_epoch % self.log_every_epochs == 0:
                # Calcualte trajectory length
                traj_len = len(trajectory["rewards"].flatten()) // self.traj_per_batch
                if self.log_trajs[1] > self.traj_per_batch:
                    self.log_trajs[1] = self.traj_per_batch
                for j_traj in range(*self.log_trajs):
                    for j_obs in range(traj_len):
                        for j_d2ch in range(self.env.delays2change_num):
                            action = trajectory['actions'].reshape(self.traj_per_batch, traj_len, -1, 2)[j_traj, j_obs, j_d2ch, :].copy()
                            # Modify step index into step
                            action[1] = self.env.step_ind_to_step(action[1])
                            new_row = {
                                'epoch': curr_epoch,
                                'trajectory': j_traj,
                                'state': (trajectory['observations'].reshape(self.traj_per_batch, -1, self.env.delays_number)[j_traj, j_obs, :] * norm_param).astype(int),
                                'action': action,
                                'reward': trajectory['rewards'].reshape(self.traj_per_batch, -1)[j_traj, j_obs],
                                'return': trajectory['returns'].reshape(self.traj_per_batch, -1)[j_traj, j_obs],
                                'aver': (np.max(trajectory['rewards'].reshape(self.traj_per_batch, -1), axis=1).mean()),
                                'std': np.sqrt(np.max(trajectory['rewards'].reshape(self.traj_per_batch, -1), axis=1).var() + 1.e-8),
                                f'ind {j_d2ch}': self.alg.distr_list[-1][j_d2ch][0].cpu().numpy().reshape(self.traj_per_batch, traj_len, -1)[j_traj, j_obs, :].tolist(),
                                f'step {j_d2ch}': self.alg.distr_list[-1][j_d2ch][1].cpu().numpy().reshape(self.traj_per_batch, traj_len, -1)[j_traj, j_obs, :].tolist()
                            }
                            self.log.loc[len(self.log)] = deepcopy(new_row)

                self.log.to_excel(os.path.join(self.save_path, "log.xlsx"), sheet_name="Training log", index=False)
            
            # Save trajectories every log_every_epochs epochs
            if curr_epoch % self.log_every_epochs == 0:
                # Save shortest trajectory, which includes best reward
                rewards = trajectory['rewards'].reshape(self.traj_per_batch, -1)
                max_inds = np.argwhere(rewards == np.max(rewards))
                ind_best = sorted(max_inds, key=lambda p: p[1])[0][0]
                # min_max_ind = np.min(max_inds)
                # ind_best = np.argwhere(min_max_ind == max_inds)[0, 0]

                # Calcualte trajectory length
                traj_len = len(trajectory["rewards"].flatten()) // self.traj_per_batch
                if self.log_trajs[1] > self.traj_per_batch:
                    self.log_trajs[1] = self.traj_per_batch
                j_traj = ind_best
                for j_obs in range(traj_len):
                    for j_d2ch in range(self.env.delays2change_num):
                        action = trajectory['actions'].reshape(self.traj_per_batch, traj_len, -1, 2)[j_traj, j_obs, j_d2ch, :].copy()
                        # Modify step index into step
                        action[1] = self.env.step_ind_to_step(action[1])
                        new_row = {
                            'epoch': curr_epoch,
                            'trajectory': j_traj,
                            'state': (trajectory['observations'].reshape(self.traj_per_batch, -1, self.env.delays_number)[j_traj, j_obs, :] * norm_param).astype(int),
                            'action': action,
                            'reward': trajectory['rewards'].reshape(self.traj_per_batch, -1)[j_traj, j_obs],
                            'return': trajectory['returns'].reshape(self.traj_per_batch, -1)[j_traj, j_obs],
                            'aver': (np.max(trajectory['rewards'].reshape(self.traj_per_batch, -1), axis=1).mean()),
                            'std': np.sqrt(np.max(trajectory['rewards'].reshape(self.traj_per_batch, -1), axis=1).var() + 1.e-8),
                            f'ind {j_d2ch}': self.alg.distr_list[-1][j_d2ch][0].cpu().numpy().reshape(self.traj_per_batch, traj_len, -1)[j_traj, j_obs, :].tolist(),
                            f'step {j_d2ch}': self.alg.distr_list[-1][j_d2ch][1].cpu().numpy().reshape(self.traj_per_batch, traj_len, -1)[j_traj, j_obs, :].tolist()
                        }
                        self.log_best_traj.loc[len(self.log_best_traj)] = deepcopy(new_row)

                self.log_best_traj.to_excel(os.path.join(self.save_path, "log_best_traj.xlsx"), sheet_name="Training log, best trajectories", index=False)

    def accum_stat(self, minibatch):

        self.summary_writer_global_step += 1

        self.alg.policy.agent.eval()
        with torch.no_grad():
            self.accum_rewards_last_mean(minibatch)
            self.accum_rewards_mean(minibatch)
            self.accum_rewards_max_mean(minibatch)
            self.accum_entropy(minibatch)
            self.accum_policy_loss(minibatch)
            self.accum_grad_norm()
            self.accum_best_perform(minibatch)
            self.accum_min_min_index_max_reward(minibatch)
            self.accum_mean_min_index_max_reward(minibatch)

            if self.alg_type == 'ppo':
                pass
                # self.accum_r2(minibatch)
                # self.accum_value_loss(minibatch)
                # self.accum_value_targets(minibatch)
                # self.accum_value_predicts(minibatch)
                # self.accum_policy_grad_norm()
                # self.accum_value_grad_norm()
                # self.accum_advantages(minibatch)
            elif self.alg_type == 'pg':
                self.accum_returns(minibatch)
                self.accum_log_policy(minibatch)

        # Flush TensorBoard summary writer
        if self.summary_writer_global_step % 10 == 0:
            self.summary_writer.flush()

    def accum_min_min_index_max_reward(self, minibatch):
        """ Calculates minimum index in batch of of minimum indices in trejectory of maximum reward among minibatch """
        rewards = minibatch["rewards"].reshape(self.traj_per_batch, -1)
        if isinstance(rewards, np.ma.MaskedArray):
            rewards = rewards.filled(-1).copy()
        min_min_ind_max_rew = np.min(np.argwhere(rewards == np.max(rewards))[:, 1])
        self.min_min_index_max_reward_list.append(min_min_ind_max_rew)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"min_min_ind_max_reward.npy"), np.ma.filled(self.min_min_index_max_reward_list, np.nan))

    def accum_mean_min_index_max_reward(self, minibatch):
        """ Calculates mean index in batch of of minimum indices in trejectory of maximum reward among minibatch """
        rewards = minibatch["rewards"].reshape(self.traj_per_batch, -1)
        if isinstance(rewards, np.ma.MaskedArray):
            rewards = rewards.filled(-1).copy()
        mean_min_index_max_rew = np.mean(np.argwhere(rewards == np.max(rewards))[:, 1])
        self.mean_min_index_max_reward_list.append(mean_min_index_max_rew)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"mean_min_index_max_reward.npy"), np.ma.filled(self.mean_min_index_max_reward_list, np.nan))

    def accum_rewards_last_mean(self, minibatch):
        """ Calculates mean of the last reward in trajectory """
        traj_len = len(minibatch["rewards"].flatten()) // self.traj_per_batch
        rewards = np.mean(minibatch["rewards"].flatten()[traj_len - 1::traj_len])
        self.rewards_last_mean.append(rewards)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards_last_mean.npy"), np.ma.filled(self.rewards_last_mean, np.nan))

    def accum_rewards_mean(self, minibatch):
        """ Calculates mean of reward in whole minibatch """
        rewards = np.mean(minibatch["rewards"].flatten())
        self.rewards_mean.append(rewards)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards_mean.npy"), np.ma.filled(self.rewards_mean, np.nan))

    def accum_rewards_max_mean(self, minibatch):
        """ Calculates mean of the max reward in trajectory """
        rewards = minibatch["rewards"].flatten()
        rewards_max_per_traj = np.max(rewards.reshape(self.traj_per_batch, -1), axis=1)
        rewards = np.mean(rewards_max_per_traj)
        self.rewards_max_mean.append(rewards)
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("train/max_reward_mean", rewards, self.summary_writer_global_step)

        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards_max_mean.npy"), np.ma.filled(self.rewards_max_mean, np.nan))
    
    def accum_rewards_max(self, minibatch):
        """ Calculates max of the last reward in trajectory """
        traj_len = len(minibatch["rewards"].flatten()) // self.traj_per_batch
        rewards = np.max(minibatch["rewards"].flatten()[traj_len - 1::traj_len])
        self.rewards_max.append(rewards)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards_max.npy"), np.ma.filled(self.rewards_max, np.nan))

    def accum_rewards_min(self, minibatch):
        """ Calculates min of the last reward in trajectory """
        traj_len = len(minibatch["rewards"].flatten()) // self.traj_per_batch
        rewards = np.min(minibatch["rewards"].flatten()[traj_len - 1::traj_len])
        self.rewards_min.append(rewards)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards_min.npy"), np.ma.filled(self.rewards_min, np.nan))

    def accum_r2(self, minibatch):
        """ R2 used to evaluate critic value prediction quality """
        with torch.no_grad():
            act = self.alg.policy.act(minibatch["observations"], training=True)
            value_predicts = act["values"].detach().cpu().numpy().flatten()
            value_targets = minibatch["value_targets"].flatten()
            ss_res = np.sum((value_targets - value_predicts) ** 2)
            ss_tot = np.sum((value_targets - np.mean(value_targets)) ** 2)
            r2_score = 1 - ss_res / ss_tot
        self.r2_score.append(r2_score)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"r2_score.npy"), np.ma.filled(self.r2_score, np.nan))

    def accum_entropy(self, minibatch):
        self.policy_entropy = self.alg.explore_loss_list
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("train/policy_entropy", self.policy_entropy[-1], self.summary_writer_global_step)
        
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"policy_entropy.npy"), np.ma.filled(self.policy_entropy, np.nan))

    def accum_value_loss(self, minibatch):
        with torch.no_grad():
            inputs = {"state": minibatch["observations"],
                      "time": minibatch["time_steps"],
                      "max_prefix": minibatch["max_prefixes"]}
            act = self.alg.policy.act(inputs, training=True)
            value_loss = self.alg.value_loss(minibatch, act).item()
            self.value_loss.append(value_loss)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_loss.npy"), np.ma.filled(self.value_loss, np.nan))

    def accum_policy_loss(self, minibatch):
        self.policy_loss = self.alg.policy_loss_list

        if self.summary_writer is not None:
            self.summary_writer.add_scalar("train/policy_loss", self.policy_loss[-1], self.summary_writer_global_step)

        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"policy_loss.npy"), np.ma.filled(self.policy_loss, np.nan))

    def accum_value_targets(self, minibatch):
        value_targets = np.mean(minibatch["value_targets"].flatten())
        self.value_targets.append(value_targets)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_targets.npy"), np.ma.filled(self.value_targets, np.nan))

    def accum_value_predicts(self, minibatch):
        with torch.no_grad():
            inputs = {"state": minibatch["observations"],
                      "time": minibatch["time_steps"],
                      "max_prefix": minibatch["max_prefixes"]}
            act = self.alg.policy.act(inputs, training=True)
            value_predicts = np.mean(act["values"].detach().cpu().numpy().flatten())
            self.value_predicts.append(value_predicts)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_predicts.npy"), np.ma.filled(self.value_predicts, np.nan))

    def accum_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("train/grad_norm", grad_norm, self.summary_writer_global_step)
        
        self.grad_norm.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm.npy"), np.ma.filled(self.grad_norm, np.nan))

    def accum_policy_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.policy_parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        self.grad_norm_policy.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm_policy.npy"), np.ma.filled(self.grad_norm_policy, np.nan))

    def accum_value_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.value_parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        self.grad_norm_value.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm_value.npy"), np.ma.filled(self.grad_norm_value, np.nan))

    def accum_advantages(self, minibatch):
        advantages = np.mean(minibatch["advantages"].flatten())
        self.advantages.append(advantages)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"advantages.npy"), np.ma.filled(self.advantages, np.nan))

    def accum_returns(self, minibatch):
        returns = minibatch["returns"].reshape(self.traj_per_batch, -1)
        returns = np.mean(returns[:, 0])
        # returns = np.mean(minibatch["returns"].flatten())
        self.returns.append(returns)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"returns.npy"), np.ma.filled(self.returns, np.nan))

    def accum_log_policy(self, minibatch):
        log_policy = np.mean(minibatch["log_probs"])
        self.log_policy.append(log_policy)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"log_policy.npy"), np.ma.filled(self.log_policy, np.nan))

    def accum_best_perform(self, minibatch):
        """
            Best performance doesn`t depend on action, so it can be
            acquired from sampled trajectory during agent interaction
            with environment
        """
        best_perform = self.env.best_perform
        self.best_perform_list.append(best_perform)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, "best_perform.npy"), np.ma.filled(self.best_perform, np.nan))
            if best_perform < self.best_perform:
                self.best_perform = best_perform
                self.best_delays = self.env.best_delays
                np.save(os.path.join(self.save_path, "best_delays.npy"), np.ma.filled(self.best_delays, np.nan))
                torch.save(self.env.tomb_raider.state_dict(), os.path.join(self.save_path, "best_params.pt"))

    def approx_kl(self, trajectory):
        with torch.no_grad():
            inputs = {"state": trajectory["observations"],
                      "time": trajectory["time_steps"],
                      "max_prefix": trajectory["max_prefixes"]}
            act = self.alg.policy.act(inputs, training=True)
            actions = torch.tensor(trajectory["actions"], device=self.alg.policy.agent.device)
            log_probs = torch.tensor(trajectory["log_probs"], device=self.alg.policy.agent.device)
            policy = act['distribution']
            delays2change_num = actions.shape[-2]
            approx_kl = 0
            for j_delay in range(delays2change_num):
                distr_ind, distr_step_ind = policy[j_delay]
                log_prob_ind = distr_ind.log_prob(actions[..., j_delay, 0])
                log_prob_step_ind = distr_step_ind.log_prob(actions[..., j_delay, 1])
                log_prob_ind_old = log_probs[..., j_delay, 0]
                log_prob_step_ind_old = log_probs[..., j_delay, 1]
                approx_kl += log_prob_ind + log_prob_step_ind - log_prob_ind_old - log_prob_step_ind_old
            approx_kl = (-1 * approx_kl / (2 * delays2change_num)).mean().item()
            self.approx_kl_list.append(approx_kl)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"approx_kl.npy"), np.ma.filled(self.approx_kl_list, np.nan))

    def clip_fraction(self, trajectory):
        eps = self.alg.cliprange_policy
        with torch.no_grad():
            inputs = {"state": trajectory["observations"],
                      "time": trajectory["time_steps"],
                      "max_prefix": trajectory["max_prefixes"]}
            act = self.alg.policy.act(inputs, training=True)
            actions = torch.tensor(trajectory["actions"], device=self.alg.policy.agent.device)
            log_probs = torch.tensor(trajectory["log_probs"], device=self.alg.policy.agent.device)
            policy = act['distribution']
            delays2change_num = actions.shape[-2]
            import_samp_ratio = 0
            for j_delay in range(delays2change_num):
                distr_ind, distr_step_ind = policy[j_delay]
                log_prob_ind = distr_ind.log_prob(actions[..., j_delay, 0])
                log_prob_step_ind = distr_step_ind.log_prob(actions[..., j_delay, 1])
                log_prob_ind_old = log_probs[..., j_delay, 0]
                log_prob_step_ind_old = log_probs[..., j_delay, 1]
                import_samp_ratio += log_prob_ind + log_prob_step_ind - log_prob_ind_old - log_prob_step_ind_old
            import_samp_ratio = torch.exp(import_samp_ratio / (2 * delays2change_num))
            clip_fraction = ((import_samp_ratio < 1.0 - eps) | (import_samp_ratio > 1.0 + eps)).float().mean().item()
            self.clip_fraction_list.append(clip_fraction)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"clip_fraction.npy"), np.ma.filled(self.clip_fraction, np.nan))


