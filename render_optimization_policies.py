import os
from os.path import exists
import numpy as np
import supersuit as ss
from array2gif import write_gif
from pettingzoo.sisl import pursuit_v4
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage,


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    return env


env = pursuit_v4.env()
env = ss.flatten_v0(env)

n_agents = 8

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath + '.zip'):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    obs_list = []
    i = 0
    env.reset()
    reward = 0

    try:
        while True:
            for agent in env.agent_iter():
                observation, reward, done, _ = env.last()
                action = (model.predict(observation, deterministic=False)[0] if not done else None)
                reward += reward

                env.step(action)
                i += 1
                if i % (len(env.possible_agents) + 1) == 0:
                    obs_list.append(
                        np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                    )

            break

        reward = reward / n_agents
        print("writing gif")
        write_gif(
            obs_list, "./optimization_gifs/" + policy + "_" + str(reward)[:5] + ".gif", fps=5
        )
    except:
        print("error")
