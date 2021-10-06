import os
from os.path import exists
import fle.flocking_env as flocking_env
import supersuit as ss
from stable_baselines3 import PPO

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
    if not exists(filepath + '.zip'):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    i = 0
    render_env.reset()

    reward = 0
    while True:
        for agent in render_env.agent_iter():
            observation, _, done, reward = render_env.last()
            action = model.predict(observation, deterministic=True)[0] if not done else None
            reward += reward
            print(action)
            render_env.step(action)

        reward = reward / 9
        print("Saving vortex logs")
        render_env.unwrapped.log_vortices("./optimization_policy_logs/" + policy +"_vortices" + ".csv")
        print("Saving bird logs")
        render_env.unwrapped.log_birds("./optimization_policy_logs/" + policy + "_birds" + ".csv")
        break
