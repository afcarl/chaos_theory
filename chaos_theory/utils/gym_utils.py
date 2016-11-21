import gym.spaces

def action_space_dim(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        return space.shape[0]