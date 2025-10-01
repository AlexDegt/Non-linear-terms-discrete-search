import numpy as np
import torch
import os, sys

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
        Calculate accumulative return taking into account discount gamma.
    """
    def __init__(self, policy, gamma=0.99):
        self.policy = policy
        self.gamma = gamma
        
    def __call__(self, trajectory):
        gamma = self.gamma

        rewards = np.asarray(trajectory["rewards"], dtype=np.float32).flatten()
        dones   = np.asarray(trajectory["resets"],  dtype=np.float32).flatten()
        T = len(rewards)
        returns = np.zeros_like(rewards, dtype=np.float32)

        R = 0.0
        for t in range(len(rewards)-1, -1, -1):
            print(len(rewards))
            # sys.exit()
            R = rewards[t] + gamma * R * (1.0 - dones[t])
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
    def __call__(self, trajectory, eps=1e-8):
        returns = np.asarray(trajectory["returns"]).flatten()
        var = np.var(returns)
        mean = np.mean(returns)
        # print((returns - mean) / np.sqrt(var + eps))
        trajectory["returns"] = (returns - mean) / np.sqrt(var + eps)

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
    """
    def __init__(self, env, alg, save_path=None, alg_type='ppo'):
        self.env = env
        self.alg = alg
        self.alg_type = alg_type
        self.save_path = save_path
        # Parameters to be tracked
        self.rewards = []
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
        self.best_perform_list = []
        self.best_perform = 100
        self.best_delays = []
        self.approx_kl_list = []
        self.clip_fraction_list = []

        self.policy_ind_distr = []
        self.policy_step_distr = []

    def save_oracle_buffer(self):
        self.env.oracle_buffer.to_excel(os.path.join(self.save_path, "oracle_buffer.xlsx"), sheet_name="Oracle buffer", index=True)
    
    def accum_stat(self, minibatch):

        self.accum_rewards(minibatch)
        self.accum_entropy(minibatch)
        self.accum_policy_loss(minibatch)
        self.accum_grad_norm()
        self.accum_best_perform(minibatch)

        if self.alg_type == 'ppo':
            self.accum_r2(minibatch)
            self.accum_value_loss(minibatch)
            self.accum_value_targets(minibatch)
            self.accum_value_predicts(minibatch)
            self.accum_policy_grad_norm()
            self.accum_value_grad_norm()
            self.accum_advantages(minibatch)
        elif self.alg_type == 'pg':
            self.accum_returns(minibatch)       

    def accum_rewards(self, minibatch):
        rewards = np.mean(minibatch["rewards"].flatten())
        self.rewards.append(rewards)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"rewards.npy"), np.asarray(self.rewards))

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
            np.save(os.path.join(self.save_path, f"r2_score.npy"), np.asarray(self.r2_score))

    def accum_entropy(self, minibatch):
        with torch.no_grad():
            act = self.alg.policy.act(minibatch["observations"], training=True)
            entropy = self.alg.explore_loss(minibatch, act).item()
            self.policy_entropy.append(entropy)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"policy_entropy.npy"), np.asarray(self.policy_entropy))

    def accum_value_loss(self, minibatch):
        with torch.no_grad():
            act = self.alg.policy.act(minibatch["observations"], training=True)
            value_loss = self.alg.value_loss(minibatch, act).item()
            self.value_loss.append(value_loss)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_loss.npy"), np.asarray(self.value_loss))

    def accum_policy_loss(self, minibatch):
        with torch.no_grad():
            act = self.alg.policy.act(minibatch["observations"], training=True)
            policy_loss = self.alg.policy_loss(minibatch, act).item()
            self.policy_loss.append(policy_loss)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"policy_loss.npy"), np.asarray(self.policy_loss))

    def accum_value_targets(self, minibatch):
        value_targets = np.mean(minibatch["value_targets"].flatten())
        self.value_targets.append(value_targets)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_targets.npy"), np.asarray(self.value_targets))

    def accum_value_predicts(self, minibatch):
        with torch.no_grad():
            act = self.alg.policy.act(minibatch["observations"], training=True)
            value_predicts = np.mean(act["values"].detach().cpu().numpy().flatten())
            self.value_predicts.append(value_predicts)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"value_predicts.npy"), np.asarray(self.value_predicts))

    def accum_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        self.grad_norm.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm.npy"), np.asarray(self.grad_norm))

    def accum_policy_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.policy_parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        self.grad_norm_policy.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm_policy.npy"), np.asarray(self.grad_norm_policy))

    def accum_value_grad_norm(self):
        with torch.no_grad():
            grads = [p.grad.detach().norm()**2 for p in self.alg.policy.agent.value_parameters() if p.grad is not None]
            grad_norm = torch.sqrt(torch.stack(grads).sum()).item()
        self.grad_norm_value.append(grad_norm)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"grad_norm_value.npy"), np.asarray(self.grad_norm_value))

    def accum_advantages(self, minibatch):
        advantages = np.mean(minibatch["advantages"].flatten())
        self.advantages.append(advantages)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"advantages.npy"), np.asarray(self.advantages))

    def accum_returns(self, minibatch):
        returns = np.mean(minibatch["returns"].flatten())
        self.returns.append(returns)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"returns.npy"), np.asarray(self.returns))

    def accum_best_perform(self, minibatch):
        """
            Best performance doesn`t depend on action, so it can be
            acquired from sampled trajectory during agent interaction
            with environment
        """
        best_perform = self.env.best_perform
        self.best_perform_list.append(best_perform)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, "best_perform.npy"), np.asarray(self.best_perform_list))
            if best_perform < self.best_perform:
                self.best_perform = best_perform
                self.best_delays = self.env.best_delays
                np.save(os.path.join(self.save_path, "best_delays.npy"), np.asarray(self.best_delays))
                torch.save(self.env.tomb_raider.state_dict(), os.path.join(self.save_path, "best_params.pt"))

    def approx_kl(self, trajectory):
        with torch.no_grad():
            act = self.alg.policy.act(trajectory["observations"], training=True)
            actions = torch.tensor(trajectory["actions"], device=self.alg.policy.agent.device)
            log_probs = torch.tensor(trajectory["log_probs"], device=self.alg.policy.agent.device)
            policy = act['distribution']
            delays2change_num = actions.shape[1]
            approx_kl = 0
            for j_delay in range(delays2change_num):
                distr_ind, distr_step_ind = policy[j_delay]
                log_prob_ind = distr_ind.log_prob(actions[:, j_delay, 0])
                log_prob_step_ind = distr_step_ind.log_prob(actions[:, j_delay, 1])
                log_prob_ind_old = log_probs[:, j_delay, 0]
                log_prob_step_ind_old = log_probs[:, j_delay, 1]
                approx_kl += log_prob_ind + log_prob_step_ind - log_prob_ind_old - log_prob_step_ind_old
            approx_kl = (-1 * approx_kl / (2 * delays2change_num)).mean().item()
            self.approx_kl_list.append(approx_kl)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"approx_kl.npy"), np.asarray(self.approx_kl_list))

    def clip_fraction(self, trajectory):
        eps = self.alg.cliprange_policy
        with torch.no_grad():
            act = self.alg.policy.act(trajectory["observations"], training=True)
            actions = torch.tensor(trajectory["actions"], device=self.alg.policy.agent.device)
            log_probs = torch.tensor(trajectory["log_probs"], device=self.alg.policy.agent.device)
            policy = act['distribution']
            delays2change_num = actions.shape[1]
            import_samp_ratio = 0
            for j_delay in range(delays2change_num):
                distr_ind, distr_step_ind = policy[j_delay]
                log_prob_ind = distr_ind.log_prob(actions[:, j_delay, 0])
                log_prob_step_ind = distr_step_ind.log_prob(actions[:, j_delay, 1])
                log_prob_ind_old = log_probs[:, j_delay, 0]
                log_prob_step_ind_old = log_probs[:, j_delay, 1]
                import_samp_ratio += log_prob_ind + log_prob_step_ind - log_prob_ind_old - log_prob_step_ind_old
            import_samp_ratio = torch.exp(import_samp_ratio / (2 * delays2change_num))
            clip_fraction = ((import_samp_ratio < 1.0 - eps) | (import_samp_ratio > 1.0 + eps)).float().mean().item()
            self.clip_fraction_list.append(clip_fraction)
        if self.save_path is not None:
            np.save(os.path.join(self.save_path, f"clip_fraction.npy"), np.asarray(self.clip_fraction_list))

    # def accum_policy_ind_distr(self, trajectory):
    #     with torch.no_grad():
    #         act = self.alg.policy.act(trajectory["observations"], training=True)
    #         policy = act['distribution']
    #         delays2change_num = actions.shape[1]
    #         approx_kl = 0
    #         for j_delay in range(delays2change_num):
    #             distr_ind, distr_step_ind = policy[j_delay]
    #             log_prob_ind = distr_ind.log_prob(actions[:, j_delay, 0])
    #             log_prob_step_ind = distr_step_ind.log_prob(actions[:, j_delay, 1])
    #             log_prob_ind_old = log_probs[:, j_delay, 0]
    #             log_prob_step_ind_old = log_probs[:, j_delay, 1]
    #             approx_kl += log_prob_ind + log_prob_step_ind - log_prob_ind_old - log_prob_step_ind_old
    #         approx_kl = (-1 * approx_kl / (2 * delays2change_num)).mean().item()
    #         self.approx_kl_list.append(approx_kl)
    #     if self.save_path is not None:
    #         np.save(os.path.join(self.save_path, f"approx_kl.npy"), np.asarray(self.approx_kl_list))