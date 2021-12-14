from stable_baselines3 import PPO
from pettingzoo.sisl import pursuit_v4
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
import numpy as np
import os
import sys
from array2gif import write_gif

os.environ["SDL_VIDEODRIVER"] = "dummy"
num = sys.argv[1]


env = pursuit_v4.env()
env = ss.flatten_v0(env)

n_agents = 8

policies = os.listdir("./mature_policies/" + str(num) + "/")

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ['a','b','c','d','e']:

        obs_list = []
        i = 0
        env.reset()
        total_reward = 0

        while True:
            for agent in env.agent_iter():
                observation, reward, done, _ = env.last()
                action = (model.predict(observation, deterministic=False)[0] if not done else None)
                total_reward += reward

                env.step(action)
                i += 1
                if i % (len(env.possible_agents) + 1) == 0:
                    obs_list.append(
                        np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                    )

            break

        total_reward = total_reward / n_agents

        if total_reward > 50:
            print("writing gif")
            write_gif(
                obs_list, "./mature_gifs/" + num + "_" + policy.split("_")[0] + j + '_' + str(total_reward)[:5] + ".gif", fps=10
            )

env.close()
