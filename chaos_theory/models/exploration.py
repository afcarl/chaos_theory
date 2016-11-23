from gym.spaces import Box
import numpy as np
import numpy.random as nr

class OUStrategy(object):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, action_space, mu=0, theta=0.15, sigma=0.2):
        assert isinstance(action_space, Box)
        assert len(action_space.shape) == 1
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = action_space
        self.state = np.ones(self.action_space.shape[0]) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_space.shape[0]) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def add_noise(self, action):
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)


class IIDStrategy(object):
    """IID Gaussian noise"""
    def __init__(self, action_space, sigma=0.2):
        assert isinstance(action_space, Box)
        assert len(action_space.shape) == 1
        self.sigma = sigma
        self.action_space = action_space
        self.reset()

    def reset(self):
        pass

    def add_noise(self, action):
        noise = np.random.randn(*action.shape) * self.sigma
        return np.clip(action + noise, self.action_space.low, self.action_space.high)

if __name__ == "__main__":
    ou = OUStrategy(Box(low=-1, high=1, shape=(1,)), mu=0, theta=0.15, sigma=0.1)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()