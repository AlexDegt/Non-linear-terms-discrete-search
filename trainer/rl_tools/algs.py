import torch
import math
import sys

class PPO:
    def __init__(self, policy, optimizer,
               cliprange_policy=0.2,
               cliprange_value=0.2,
               value_loss_coef=0.25,
               explore_loss_coef=0.1,
               max_grad_norm_policy=0.5,
               max_grad_norm_value=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange_policy = cliprange_policy
        self.cliprange_value = cliprange_value
        self.value_loss_coef = value_loss_coef
        self.explore_loss_coef = explore_loss_coef
        self.max_grad_norm_policy = max_grad_norm_policy
        self.max_grad_norm_value = max_grad_norm_value

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        actions = torch.tensor(trajectory["actions"], device=self.policy.agent.device)
        log_probs = torch.tensor(trajectory["log_probs"], device=self.policy.agent.device)
        advantages = torch.tensor(trajectory["advantages"], device=self.policy.agent.device)
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
        loss_per_sample = import_samp_ratio * advantages
        # import_samp_ratio_clip = import_samp_ratio
        import_samp_ratio_clip = torch.clamp(import_samp_ratio, 1 - self.cliprange_policy, 1 + self.cliprange_policy)
        loss_per_sample_clip = import_samp_ratio_clip * advantages
        return -1 * torch.mean(torch.min(loss_per_sample, loss_per_sample_clip))

    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        value_targets = torch.tensor(trajectory["value_targets"], device=self.policy.agent.device)
        values_old = torch.tensor(trajectory["values"], device=self.policy.agent.device).flatten()
        values = act['values']
        squares_vector_simple = (values - value_targets) ** 2
        value_pred_clipped = values_old + (values - values_old).clamp(-self.cliprange_value, self.cliprange_value)
        squares_vector_clip = (value_pred_clipped - value_targets) ** 2
        # squares_vector_clip = squares_vector_simple
        return torch.mean(torch.max(squares_vector_simple, squares_vector_clip))

    def explore_loss(self, trajectory, act):
        """ Computes policy entropy on a given trajectory. """
        actions = torch.tensor(trajectory["actions"], device=self.policy.agent.device)
        policy = act['distribution']
        entropy = 0
        for distr_ind, distr_step_ind in policy:
            entropy += distr_ind.entropy().mean() + distr_step_ind.entropy().mean()
        return entropy

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)
        explore_loss = self.explore_loss(trajectory, act)
        return policy_loss + self.value_loss_coef * value_loss - self.explore_loss_coef * explore_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()

        loss = self.loss(trajectory)
        loss.backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.agent.parameters(), self.max_grad_norm)
        grad_norm_policy = torch.nn.utils.clip_grad_norm_(self.policy.agent.policy_parameters(), self.max_grad_norm_policy)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(self.policy.agent.value_parameters(), self.max_grad_norm_value)

        self.optimizer.step()

    def grad_norm(self, params):
        sqsum = 0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            sqsum += g.pow(2).sum().item()
        return math.sqrt(sqsum + 1e-18)

    def debug_MLPSharedBackPolicy(self):
        # Debug shared-neck arhitecture
        shared_params = [self.policy.agent.layer_inp.weight]
        shared_params.extend([p.weight for p in self.policy.agent.hidden_shared])

        policy_params = []
        policy_params.extend([p.weight for p in self.policy.agent.hidden_policy])
        policy_params.extend([p.weight for p in self.policy.agent.delay_range])
        policy_params.extend([p.weight for p in self.policy.agent.delay_steps])

        value_params = [self.policy.agent.value_out.weight]
        value_params.extend([p.weight for p in self.policy.agent.hidden_value])

        act = self.policy.act(trajectory["observations"], training=True)

        # Gradient for value head
        self.optimizer.zero_grad(set_to_none=True)
        L_pi = self.policy_loss(trajectory, act)
        L_v = self.value_loss(trajectory, act)
        L_ent = self.explore_loss(trajectory, act)

        (L_v * self.value_loss_coef).backward(retain_graph=True)
        gn_shared_v = self.grad_norm(shared_params)
        gn_value = self.grad_norm(value_params)
        gn_policy = self.grad_norm(policy_params)

        # Gradient for policy head
        self.optimizer.zero_grad(set_to_none=True)
        L_pi = self.policy_loss(trajectory, act)
        L_v = self.value_loss(trajectory, act)
        L_ent = self.explore_loss(trajectory, act)

        (L_pi - L_ent * self.explore_loss_coef).backward(retain_graph=True)
        gn_shared_p = self.grad_norm(shared_params)
        gn_policy2 = self.grad_norm(policy_params)
        gn_value2 = self.grad_norm(value_params)

        # Gradients for whole loss
        self.optimizer.zero_grad(set_to_none=True)
        L_pi = self.policy_loss(trajectory, act)
        L_v = self.value_loss(trajectory, act)
        L_ent = self.explore_loss(trajectory, act)

        (L_pi + L_v * self.value_loss_coef - L_ent * self.explore_loss_coef).backward(retain_graph=True)
        gn_shared_all = self.grad_norm(shared_params)
        gn_total = self.grad_norm(shared_params + policy_params + value_params)

        print(f"grad L_value w.r.t. shared = {gn_shared_v}\n",
              f"grad L_value w.r.t. value = {gn_value}\n" ,
              f"grad L_value w.r.t. policy = {gn_policy}\n",
              f"grad L_policy w.r.t. shared = {gn_shared_p}\n",
              f"grad L_policy w.r.t. value = {gn_value2}\n" ,
              f"grad L_policy w.r.t. policy = {gn_policy2}\n",
              f"grad L_all w.r.t. shared = {gn_shared_all}\n",
              f"grad L_all w.r.t. all params = {gn_total}")

class PolicyGradient:
    def __init__(self, policy, optimizer,
               explore_loss_coef=0.1,
               max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.explore_loss_coef = explore_loss_coef
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        actions = torch.tensor(trajectory["actions"], device=self.policy.agent.device)
        returns = torch.tensor(trajectory["returns"], device=self.policy.agent.device)
        policy = act['distribution']
        delays2change_num = actions.shape[1]
        log_policy = 0
        for j_delay in range(delays2change_num):
            distr_ind, distr_step_ind = policy[j_delay]
            log_prob_ind = distr_ind.log_prob(actions[:, j_delay, 0])
            log_prob_step_ind = distr_step_ind.log_prob(actions[:, j_delay, 1])
            log_policy += log_prob_ind + log_prob_step_ind
        # log_policy /= (2 * delays2change_num)
        return -1 * torch.mean(log_policy * returns)

    def explore_loss(self, trajectory, act):
        """ Computes policy entropy on a given trajectory. """
        actions = torch.tensor(trajectory["actions"], device=self.policy.agent.device)
        policy = act['distribution']
        entropy = 0
        for distr_ind, distr_step_ind in policy:
            entropy += distr_ind.entropy().mean() + distr_step_ind.entropy().mean()
        return entropy

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        explore_loss = self.explore_loss(trajectory, act)
        return policy_loss - self.explore_loss_coef * explore_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        
        loss = self.loss(trajectory)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.agent.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def grad_norm(self, params):
        sqsum = 0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            sqsum += g.pow(2).sum().item()
        return math.sqrt(sqsum + 1e-18)