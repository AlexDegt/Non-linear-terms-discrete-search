import torch

class PPO:
    def __init__(self, policy, optimizer,
               cliprange=0.2,
               value_loss_coef=0.25,
               explore_loss_coef=0.1,
               max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.explore_loss_coef = explore_loss_coef
        self.max_grad_norm = max_grad_norm

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
        import_samp_ratio = torch.exp(import_samp_ratio)
        loss_per_sample = import_samp_ratio * advantages
        import_samp_ratio_clip = import_samp_ratio
        # import_samp_ratio_clip = torch.clamp(import_samp_ratio, 1 - self.cliprange, 1 + self.cliprange)
        loss_per_sample_clip = import_samp_ratio_clip * advantages
        return -1 * torch.mean(torch.min(loss_per_sample, loss_per_sample_clip))

    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        value_targets = torch.tensor(trajectory["value_targets"], device=self.policy.agent.device)
        values_old = torch.tensor(trajectory["values"], device=self.policy.agent.device).flatten()
        values = act['values']
        squares_vector_simple = (values - value_targets) ** 2
        # value_pred_clipped = values_old + (values - values_old).clamp(-self.cliprange, self.cliprange)
        # squares_vector_clip = (value_pred_clipped - value_targets) ** 2
        squares_vector_clip = squares_vector_simple
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

        self.optimizer.step()