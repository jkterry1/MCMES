import os
from os.path import exists

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from PIL import Image
from stable_baselines3 import PPO

n_agents = 4

env = knights_archers_zombies_v10.env()
env = ss.black_death_v3(env)

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath + ".zip"):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    video_log = []
    i = 0
    env.reset()
    total_reward = 0

    try:
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

        print("writing gif")
        video_log[0].save(
            "./optimization_gifs/" + policy + "_" + str(total_reward)[:5] + ".gif",
            save_all=True,
            append_images=video_log[1:],
            optimize=False,
            duration=int(1000 / 15),
            loop=0,
        )
    except:
        print("error")
