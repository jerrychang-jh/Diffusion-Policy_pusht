import gymnasium as gym
import gym_pusht

# env = gym.make("gym_pusht/PushT-v0")
# print(env.observation_space)
# print(env.action_space)

from lerobot.envs.factory import make_env
from lerobot.envs.configs import PushtEnv
import numpy as np

env_map = make_env(PushtEnv())
tg = next(iter(env_map))
tid = next(iter(env_map[tg]))
env = env_map[tg][tid]

obs, info = env.reset()
print("env obs type:", type(obs))
if isinstance(obs, dict):
    print("env keys:", obs.keys())
    for k in sorted(obs.keys()):
        v = obs[k]
        if hasattr(v, "shape"):
            print(k, np.asarray(v).shape, np.asarray(v).dtype)
