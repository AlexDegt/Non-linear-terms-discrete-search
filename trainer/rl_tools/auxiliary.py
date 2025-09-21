import numpy as np

class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        
    def __call__(self, trajectory):
        gamma = self.gamma
        lam = self.lambda_

        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
        values  = np.asarray(trajectory["values"],  dtype=np.float32)
        dones   = np.asarray(trajectory["resets"],  dtype=np.float32)

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