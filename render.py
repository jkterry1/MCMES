from stable_baselines3 import PPO
import supersuit as ss
import numpy as np
import os
import sys
from array2gif import write_gif
import sumo_rl

#import pyglet

#pyglet.options['headless'] = True
num = sys.argv[1]


# def image_transpose(env):
#     if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#         env = VecTransposeImage(env)
#     return env

RESOLUTION = (5000, 5000)

env = sumo_rl.ingolstadt7(sumo_warnings=False, virtual_display=RESOLUTION)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)

policies = os.listdir("./mature_policies/" + str(num) + "/")

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    obs_list = []
    i = 0
    env.reset()

    while True:
        for agent in env.agent_iter():
            observation, _, done, _ = env.last()
            action = (
                model.predict(observation, deterministic=True)[0] if not done else None
            )

            env.step(action)
            i += 1
            if i % (len(env.possible_agents) + 1) == 0:
                obs_list.append(
                    np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                )
        env.close()
        break

    print("writing gif")
    write_gif(
        obs_list, "./mature_gifs/" + num + "_" + policy.split(".")[0] + ".gif", fps=50
    )
