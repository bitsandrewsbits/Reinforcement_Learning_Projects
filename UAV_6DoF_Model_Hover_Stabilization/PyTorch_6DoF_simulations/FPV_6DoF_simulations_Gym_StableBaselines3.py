# simulations using SAC from Stable-Baseline-v3
import torch
import gymnasium as gym
from stable_baselines3 import SAC

import UAV_6DoF_model_adapter_for_Gymnasium_Env as adpt_gym_env

env = adpt_gym_env.UAV_6DoF_Adapter_For_GymEnv()

model = SAC(
    "MlpPolicy", env,
    verbose = 1, device = "cuda"
)

model.learn(total_timesteps = 10000, log_interval = 4)
model.save("sac_6DoF_hover_stabilization")

observation, info = env.reset()
current_episode_step = 0
while current_episode_step < 200:
	action, _states = model.predict(observation, deterministic = True)
	observation, reward, terminated, truncated, info = env.step(action)
	print("Observation:", observation)
	print("Reward:", reward)
	print("Terminated:", terminated)
	print("Truncated:", truncated)
	print("info:", info)
	print("-" * 20)
	
	if terminated or truncated:
		obs, info = env.reset()
	current_episode_step += 1
