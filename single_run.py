import json
import sys
import time

import fle.flocking_env as flocking_env
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecTransposeImage
from torch import nn as nn

n_evaluations = 20
n_agents = 9
n_envs = 4
total_energy_j = 46000
total_distance_m = 870
hz = 500
crash_reward = -10
episodes = 2000
nerve_impulse_hz = 200
reaction_frames = 0
n_timesteps = hz * 60 * n_agents * episodes
distance_reward_per_m = 100 / total_distance_m
energy_reward_per_j = -10 / total_energy_j
skip_frames = int(hz / nerve_impulse_hz)

env = flocking_env.parallel_env(
    N=n_agents,
    h=1 / hz,
    energy_reward=energy_reward_per_j,
    forward_reward=distance_reward_per_m,
    crash_reward=crash_reward,
    LIA=True,
)
env = ss.delay_observations_v0(env, reaction_frames)
env = ss.frame_skip_v0(env, skip_frames)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)

eval_env = flocking_env.parallel_env(
    N=n_agents,
    h=1 / hz,
    energy_reward=energy_reward_per_j,
    forward_reward=distance_reward_per_m,
    crash_reward=crash_reward,
    LIA=True,
)
eval_env = ss.delay_observations_v0(eval_env, reaction_frames)
eval_env = ss.frame_skip_v0(eval_env, skip_frames)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class="stable_baselines3")
eval_env = VecMonitor(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

model = PPO("MlpPolicy", env, verbose=3)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./single_run_logs",
    log_path="./single_run_logs",
    eval_freq=eval_freq,
    deterministic=True,
    render=False,
)

start = time.time()
model.learn(total_timesteps=n_timesteps, callback=eval_callback)


end = time.time()

print("Learning run time (seconds) was " + str(end - start))
