import sys
import json
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
from torch import nn as nn

num = sys.argv[1]
n_evaluations = 20
n_agents = 3
n_envs = 4
n_timesteps = 8000000

with open("./hyperparameter_jsons/" + "hyperparameters_" + num + ".json") as f:
    params = json.load(f)

print(params)


net_arch = {"small": [dict(pi=[64, 64], vf=[64, 64])], "medium": [dict(pi=[256, 256], vf=[256, 256])], "large": [dict(pi=[400, 300], vf=[400, 300])], "extra_large": [dict(pi=[750, 750, 500], vf=[750, 750, 500])]}[params["net_arch"]]
activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[params["activation_fn"]]
ortho_init = False

params["policy_kwargs"] = dict(net_arch=net_arch, activation_fn=activation_fn, ortho_init=ortho_init)

del params["net_arch"]
del params["activation_fn"]


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    return env


env = multiwalker_v7.parallel_env()
env = ss.normalize_obs_v0(env)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)
env = image_transpose(env)

eval_env = multiwalker_v7.parallel_env()
eval_env = ss.normalize_obs_v0(eval_env)
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(
    eval_env, 1, num_cpus=1, base_class="stable_baselines3"
)
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

all_mean_rewards = []
for i in range(10):
    try:
        model = PPO("MlpPolicy", env, verbose=3, **params)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./eval_logs/" + num + "/",
            log_path="./eval_logs/" + num + "/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        model.learn(total_timesteps=n_timesteps, callback=eval_callback)
        model = PPO.load("./eval_logs/" + num + "/" + "best_model")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, deterministic=True, n_eval_episodes=25
        )
        print(mean_reward)
        print(std_reward)
        all_mean_rewards.append(mean_reward)
        if mean_reward > 90:
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
        print("error")

if len(all_mean_rewards) > 0:
    print(sum(all_mean_rewards) / len(all_mean_rewards))
else:
    print("No mature policies found")
