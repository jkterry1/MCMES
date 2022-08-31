import os
import sys

import numpy as np
import supersuit as ss
from array2gif import write_gif
from pettingzoo.sisl import multiwalker_v9
from stable_baselines3 import PPO

num = sys.argv[1]


env = multiwalker_v9.env()
env = ss.frame_stack_v1(env, 3)

n_agents = 3

policies = os.listdir("./mature_policies/" + str(num) + "/")

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ["a", "b", "c", "d", "e"]:

        obs_list = []
        i = 0
        env.reset()
        total_reward = 0

        while True:
            for agent in env.agent_iter():
                observation, reward, done, _ = env.last()
                action = (
                    model.predict(observation, deterministic=True)[0]
                    if not done
                    else None
                )
                total_reward += reward

                env.step(action)
                i += 1
                if i % (len(env.possible_agents) + 1) == 0:
                    obs_list.append(
                        np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                    )

            break

        total_reward = total_reward / n_agents

        if total_reward > 70:
            print("writing gif")
            write_gif(
                obs_list,
                "./mature_gifs/"
                + num
                + "_"
                + policy.split("_")[0]
                + j
                + "_"
                + str(total_reward)[:5]
                + ".gif",
                fps=50,
            )
