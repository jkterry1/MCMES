import sys
import json
from stable_baselines3 import PPO
import fle.flocking_env as flocking_env
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)

num = sys.argv[1]
n_evaluations = 20
n_agents = 4
n_envs = 4
n_timesteps = 10000000

with open("./hyperparameter_jsons/" + "hyperparameters_" + num + ".json") as f:
    params = json.load(f)

print(params)


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    return env


<<<<<<< HEAD
def create_envs(n_envs: int, eval_env: bool = False, no_log: bool = False):

    n_agents = 9
    total_energy_j = 24212
    total_distance_m = 894
    hz = 500
    crash_reward = -10
    nerve_impulse_hz = 200
    reaction_frames = 0
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
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)

    env = self._maybe_normalize(env, eval_env)

    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        if self.verbose > 0:
            print("Wrapping into a VecTransposeImage")
        env = VecTransposeImage(env)

    return env

env = create_envs(n_envs)
eval_env = create_envs(n_envs, eval_env=True)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

all_mean_rewards = []

for i in range(10):
    try:
        model = PPO("CnnPolicy", env, verbose=1, **params)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./eval_logs/" + num + "/" + str(i) + "/",
            log_path="./eval_logs/" + num + "/" + str(i) + "/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        model.learn(total_timesteps=n_timesteps, callback=eval_callback)
        model = PPO.load("./eval_logs/" + num + "/" + str(i) + "/" + "best_model")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, deterministic=True, n_eval_episodes=25
        )
        print(mean_reward)
        print(std_reward)
        all_mean_rewards.append(mean_reward)
        if mean_reward > 1:
            model.save(
                "./mature_policies/"
                + str(num)
                + "/"
                + str(i)
                + "_"
                + str(mean_reward).split(".")[0]
                + ".zip"
            )
    except:
        print("Error occurred during evaluation")

if len(all_mean_rewards) > 0:
    print(sum(all_mean_rewards) / len(all_mean_rewards))
else:
    print("No mature policies found")
