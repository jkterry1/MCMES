import os
from os.path import exists

import numpy as np
import supersuit as ss
from array2gif import write_gif
from social_dilemmas.envs import pettingzoo_env
from stable_baselines3 import PPO

env_name = "harvest"
n_agents = 5
num_frames = 4

env = pettingzoo_env.env(
    env=env_name,
    num_agents=n_agents,
)
env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
env = ss.frame_stack_v1(env, num_frames)

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath + ".zip"):
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
                action = (
                    model.predict(observation, deterministic=True)[0]
                    if not done
                    else None
                )
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
            obs_list, "./optimization_gifs/" + policy + "_" + "reward" + ".gif", fps=5
        )
    except:
        print("error")
