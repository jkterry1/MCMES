from stable_baselines3 import PPO
from pettingzoo.butterfly import knights_archers_zombies_v9
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
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"
num = sys.argv[1]


# def image_transpose(env):
#     if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#         env = VecTransposeImage(env)
#     return env


env = knights_archers_zombies_v9.env()
env = ss.black_death_v3(env)

policies = os.listdir("./mature_policies/" + str(num) + "/")

n_agents = 4

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ["a", "b", "c", "d", "e"]:

        print(f"rendering, {j}")

        video_log = []
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
                    video_log.append(Image.fromarray(env.render(mode="rgb_array")))

            break

        total_reward = total_reward / n_agents

        if total_reward > 3:
            print("writing gif")

            video_log[0].save(
                "./mature_gifs/"
                + num
                + "_"
                + policy.split("_")[0]
                + j
                + "_"
                + str(total_reward)[:5]
                + ".gif",
                save_all=True,
                append_images=video_log[1:],
                optimize=False,
                duration=int(1000 / 15),
                loop=0,
            )
