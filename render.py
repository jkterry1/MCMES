import os
import sys

import fle.flocking_env as flocking_env
import supersuit as ss
from stable_baselines3 import PPO

num = sys.argv[1]

n_agents = 9
n_envs = 4
total_energy_j = 24164
total_distance_m = 894
hz = 500
crash_reward = -10
episodes = 12000
nerve_impulse_hz = 200
reaction_frames = 0
time = 60
n_timesteps = hz * time * n_agents * episodes
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
render_env = ss.frame_stack_v1(render_env, 4)

policies = os.listdir("./mature_policies/" + str(num) + "/")

for policy in policies:
    print("Loading new policy")
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    i = 0
    render_env.reset()

    while True:
        for agent in render_env.agent_iter():
            observation, _, done, _ = render_env.last()
            action = model.predict(observation, deterministic=True)[0] if not done else None
            render_env.step(action)

        print("Saving vortex logs")
        render_env.unwrapped.log_vortices("./mature_simulations/" + num + "_" + policy.split(".")[0] + "_vortices" + ".csv")
        print("Saving bird logs")
        render_env.unwrapped.log_birds("./mature_simulations/" + num + "_" + policy.split(".")[0] + "_birds" + ".csv")
        break
