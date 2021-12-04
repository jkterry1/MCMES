from stable_baselines3 import PPO
from pettingzoo.butterfly import cooperative_pong_v4
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


# def image_transpose(env):
#     if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#         env = VecTransposeImage(env)
#     return env


env = cooperative_pong_v4.parallel_env()
player1 = env.possible_agents[0]


def invert_agent_indication(obs, agent):
    if len(obs.shape) == 2:
        obs = obs.reshape(obs.shape + (1,))
    obs2 = obs if agent == player1 else 255 - obs
    return np.concatenate([obs, obs2], axis=2)


env = cooperative_pong_v4.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.observation_lambda_v0(env, invert_agent_indication)
env = ss.frame_stack_v1(env, 3)


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
        obs_list, "./mature_gifs/" + num + "_" + policy.split(".")[0] + ".gif", fps=15
    )
