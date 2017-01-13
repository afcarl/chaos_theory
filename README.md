# chaos_theory

Implementations of various RL algorithms using OpenAI Gym environments
- [REINFORCE] (https://github.com/justinjfu/chaos_theory/blob/master/chaos_theory/algorithm/reinforce.py)
- [DDPG] (https://github.com/justinjfu/chaos_theory/blob/master/chaos_theory/algorithm/ddpg_old.py)
- NAF

TODO:
- Cross-Entropy Method
- TRPO/Natural Policy Gradients
- QProp


Requirements
============

- Python 2.7
- [OpenAI Gym] (https://github.com/openai/gym)
- [Tensorflow] (https://github.com/tensorflow/tensorflow)
- Numpy/Scipy

Optional Libraries
------------
- imageio (for recording gifs)

Instructions
============

<img src="https://github.com/justinjfu/resources/blob/master/rl_site/cheetah_ddpg.gif" width="250">
<img src="https://github.com/justinjfu/resources/blob/master/rl_site/cartpole_reinf.gif" width="250">

REINFORCE Cartpole experiment:
```
python scripts/run_reinforce.py
````

DDPG Half-Cheetah experiment:
```
python scripts/run_ddpg2.py
````

