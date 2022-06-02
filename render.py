import os
import sys
import numpy as np
import supersuit as ss
from array2gif import write_gif
from pettingzoo.butterfly import cooperative_pong_v4
from stable_baselines3 import PPO

os.environ["SDL_VIDEODRIVER"] = "dummy"
num = sys.argv[1]


env = cooperative_pong_v4.parallel_env()
player1 = env.possible_agents[0]


def invert_agent_indication(obs, agent):
    if len(obs.shape) == 2:
        obs = obs.reshape(obs.shape + (1,))
    obs2 = obs if agent == player1 else 255 - obs
    return np.concatenate([obs, obs2], axis=2)


n_agents = 2

env = cooperative_pong_v4.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.observation_lambda_v0(env, invert_agent_indication)
env = ss.frame_stack_v1(env, 3)


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
                action = model.predict(observation, deterministic=True)[0] if not done else None
                total_reward += reward

                env.step(action)
                i += 1
                if i % (len(env.possible_agents) + 1) == 0:
                    obs_list.append(np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2)))

            break

        total_reward = total_reward / n_agents

        if total_reward > 90:
            print("writing gif")
            write_gif(
                obs_list,
                "./mature_gifs/" + num + "_" + policy.split("_")[0] + j + "_" + str(total_reward)[:5] + ".gif",
                fps=15,
            )
