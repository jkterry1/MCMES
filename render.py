import os
import subprocess
import sys

import imageio
import sumo_rl
import supersuit as ss
from pettingzoo.utils.conversions import from_parallel
from stable_baselines3 import PPO

num = sys.argv[1]


RESOLUTION = (7000, 7000)  # required for cars to be visible

n_agents = 21

env = sumo_rl.ingolstadt21(
    sumo_warnings=False, virtual_display=RESOLUTION, use_gui=True
)
env = from_parallel(env)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)

policies = os.listdir("./mature_policies/" + str(num) + "/")

cache = "./render_cache/" + str(num) + "/"

os.mkdir(cache)

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ["a", "b", "c", "d", "e"]:

        i = 0
        k = 0
        env.reset()
        total_reward = 0

        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            action = (
                model.predict(observation, deterministic=True)[0] if not done else None
            )
            total_reward += reward

            env.step(action)
            i += 1
            if i % (len(env.possible_agents) + 1) == 0:
                render_array = env.render(mode="rgb_array")
                imageio.imwrite(cache + str(k) + ".png", render_array.astype("uint8"))
                k = k + 1

        total_reward = total_reward / n_agents

        if total_reward > -0.2:
            print("Rendering frames")
            name = (
                "./mature_gifs/"
                + num
                + "_"
                + policy.split("_")[0]
                + j
                + "_"
                + str(total_reward)[:5]
                + ".mp4"
            )
            subprocess.run(
                ["ffmpeg", "-y", "-framerate", "5", "-i", cache + "%d.png", name]
            )

        # clear scratch directory
        for file in os.scandir(cache):
            os.remove(file.path)

env.close()
