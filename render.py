from stable_baselines3 import PPO
from pettingzoo.sisl import multiwalker_v7
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

#import pyglet

#pyglet.options['headless'] = True
num = sys.argv[1]


# def image_transpose(env):
#     if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
#         env = VecTransposeImage(env)
#     return env


env = multiwalker_v7.env()
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
        obs_list, "./mature_gifs/" + num + "_" + policy.split(".")[0] + ".gif", fps=50
    )


import gym

env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1', force=True)
observation = env.reset()
for t in range(100):
#    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.monitor.close()