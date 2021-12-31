from stable_baselines3 import PPO
import supersuit as ss
import numpy as np
import os
import sys
import sumo_rl
from pettingzoo.utils.conversions import from_parallel
import imageio
import subprocess

#import pyglet

#pyglet.options['headless'] = True
num = sys.argv[1]


# def image_transpose(env):
#     if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#         env = VecTransposeImage(env)
#     return env

RESOLUTION = (4000, 4000)

n_agents = 7

env = sumo_rl.ingolstadt7(sumo_warnings=False, virtual_display=RESOLUTION, use_gui=True)
env = from_parallel(env)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)

policies = os.listdir("./mature_policies/" + str(num) + "/")

cache = "./render_cache/" + str(num) + "/"

os.mkdir(cache)

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ['a', 'b', 'c', 'd', 'e']:

        i = 0
        k = 0
        env.reset()
        total_reward = 0

        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            action = (model.predict(observation, deterministic=True)[0] if not done else None)
            total_reward += reward

            env.step(action)
            i += 1
            if i % (len(env.possible_agents) + 1) == 0:
                render_array = env.render(mode="rgb_array")
                imageio.imwrite(cache + str(k) + '.jpg', render_array.astype('uint8'))
                k = k + 1

        total_reward = total_reward / n_agents

        if total_reward > -.1:
            print("Rendering frames")
            name = "./mature_gifs/" + num + "_" + policy.split("_")[0] + j + '_' + str(total_reward)[:5] + ".mp4"
            subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", cache + "%d.jpg", name])

        # clear scratch directory
        for file in os.scandir(cache):
            os.remove(file.path)

env.close()
