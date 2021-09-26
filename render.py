import os
import sys
from os.path import exists

import fle.flocking_env as flocking_env
import numpy as np
import supersuit as ss
from array2gif import write_gif
from pettingzoo.butterfly import pistonball_v4
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecTransposeImage

#num = sys.argv[1]

n_evaluations = 20
n_agents = 9
n_envs = 4
total_energy_j = 46000
total_distance_m = 870
hz = 500
crash_reward = -10
episodes = 300
nerve_impulse_hz = 200
reaction_frames = 0
n_timesteps = hz * 60 * n_agents * episodes
distance_reward_per_m = 100 / total_distance_m
energy_reward_per_j = -10 / total_energy_j
skip_frames = int(hz / nerve_impulse_hz)


render_env = flocking_env.env(
    N=n_agents,
    h=1 / hz,
    energy_reward=energy_reward_per_j,
    forward_reward=distance_reward_per_m,
    crash_reward=crash_reward,
    LIA=True,
)
render_env = ss.delay_observations_v0(render_env, reaction_frames)
render_env = ss.frame_skip_v0(render_env, skip_frames)

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath+'.zip'):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    i = 0
    render_env.reset()

    while True:
        for agent in render_env.agent_iter():
            observation, _, done, _ = render_env.last()
            action = model.predict(observation, deterministic=True)[0] if not done else None
            print(action)
            render_env.step(action)

        print("Saving vortex logs")
        render_env.unwrapped.log_vortices("./results/" + policy +"_vortices" + ".csv")
        print("Saving bird logs")
        render_env.unwrapped.log_birds("./results/" + policy + "_birds" + ".csv")
        break
