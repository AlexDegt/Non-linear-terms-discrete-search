class PPO:
    def __init__(self, policy, optimizer,
               cliprange=0.2,
               value_loss_coef=0.25,
               max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        # Note that we don't need entropy regularization for this env.
        self.max_grad_norm = max_grad_norm

    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        pass
        # < insert your code here >

    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        pass
        # < insert your code here >

    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)

        return policy_loss + self.value_loss_coef * value_loss

    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        loss = self.loss(trajectory)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)

        self.optimizer.step()