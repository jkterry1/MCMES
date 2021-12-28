import json
import sys

import gym
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from social_dilemmas.envs import pettingzoo_env

num = sys.argv[1]
n_evaluations = 20
n_agents = 5
n_cpus = 4
n_envs = 8
n_timesteps = 1e7
env_name = "harvest"

with open("./hyperparameter_jsons/" + "hyperparameters_" + num + ".json") as f:
    params = json.load(f)

print(params)

num_frames = params["num_frames"] if "num_frames" in params else 4


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128, # placeholder
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
        activation_fn=F.relu,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(
            in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1]
        )
        self.activation_fn = activation_fn
        self.num_frames = num_frames

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        if self.num_frames > 4:
            observations = observations.permute(0, 3, 1, 2)
        features = self.activation_fn(self.conv(observations))
        features = torch.flatten(features, start_dim=1)
        features = self.activation_fn(self.fc1(features))
        features = self.activation_fn(self.fc2(features))
        return features


activation_fn = {"tanh": F.tanh, "relu": F.relu, "elu": F.elu, "leaky_relu": F.leaky_relu}[params["activation_fn"]]
net_arch = {"small": [dict(pi=[64, 64], vf=[64, 64])], "medium": [dict(pi=[256, 256], vf=[256, 256])], "large": [dict(pi=[400, 300], vf=[400, 300])], "extra_large": [dict(pi=[750, 750, 500], vf=[750, 750, 500])]}[params["net_arch"]]
# fcnet_hiddens = {"small": [128, 32], "medium": [256, 64], "large": [1024, 128], "extra_large": [1024, 256]}[params["net_arch"]]
fcnet_hiddens = [1024, 128]

params["policy_kwargs"] = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        num_frames=num_frames,
        fcnet_hiddens=fcnet_hiddens,
        activation_fn=activation_fn
    ),
    net_arch=net_arch,
)

del params["net_arch"]
del params["activation_fn"]

env = pettingzoo_env.parallel_env(
    env=env_name,
    num_agents=n_agents,
)
env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
env = ss.frame_stack_v1(env, num_frames)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=n_cpus, base_class="stable_baselines3")
env = VecTransposeImage(env)
env = VecMonitor(env)

eval_env = pettingzoo_env.parallel_env(
    env=env_name,
    num_agents=n_agents,
)
eval_env = ss.observation_lambda_v0(eval_env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
eval_env = ss.frame_stack_v1(eval_env, num_frames)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(
    eval_env, 1, num_cpus=n_cpus, base_class="stable_baselines3"
)
eval_env = VecTransposeImage(eval_env)
eval_env = VecMonitor(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

all_mean_rewards = []

for i in range(10):
    model = PPO("CnnPolicy", env, verbose=1, **params)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./eval_logs/" + num + "/" + str(i) + "/",
        log_path="./eval_logs/" + num + "/" + str(i) + "/",
        eval_freq=eval_freq,
        deterministic=False,
        render=False,
    )
    model.learn(total_timesteps=n_timesteps, callback=eval_callback)
    model = PPO.load("./eval_logs/" + num + "/" + str(i) + "/" + "best_model")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, deterministic=False, n_eval_episodes=25
    )
    print(mean_reward)
    print(std_reward)
    all_mean_rewards.append(mean_reward)
    if mean_reward > 0:
        model.save(
            "./mature_policies/"
            + str(num)
            + "/"
            + str(i)
            + "_"
            + str(mean_reward).split(".")[0]
            + ".zip"
        )

if len(all_mean_rewards) > 0:
    print(sum(all_mean_rewards) / len(all_mean_rewards))
else:
    print("No mature policies found")
