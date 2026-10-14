# Drone environment(6DoF model) class as Gymnasium API
import torch
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
import numpy as np

import UAV_6DoF_model_environment as uav_env
import additional_functions as adf

class UAV_6DoF_Adapter_For_GymEnv(gym.Env):
	def __init__(self,
	hover_stabilization_height_in_m: float, hover_time_in_sec: float,
	external_drag_disturbance: bool, external_wind_disturbance: bool,
	simulation_max_steps: int, level_of_params_uncertanty = "delta_0",
	wind_gust_disturbance_time_level = "low",
	DR_mode = True):
		super().__init__()
		self.hover_stabilization_height_in_m = hover_stabilization_height_in_m
		self.hover_time_in_sec = hover_time_in_sec
		self.simulation_max_steps = simulation_max_steps
		self.external_drag_disturbance = external_drag_disturbance
		self.external_wind_disturbance = external_wind_disturbance
		self.level_of_params_uncertanty = level_of_params_uncertanty
		self.wind_gust_disturbance_time_level = wind_gust_disturbance_time_level
		self.DR_mode = DR_mode
		self.uav_6dof_env = uav_env.UAV_6DoF_Model_Environment(
			uav_action_name = 'hover_stabilization',
			discount_gamma = 0.99,
			reward_coefs = {"alpha": 2.5, "beta": 0.4, "gamma": 1.2, "delta": 0.05},
			init_state_vector = {
				"p": np.array([0.0, 0.0, 0.0]),
				"v": np.array([0, 0, 0]),
				"q": np.array([1, 0, 0, 0]),
				"omega": np.array([0, 0, 0])
			},
			state_space_restrictions = {
				"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
				"angeles": {"fi": [-45, 45], "teta": [-45, 45], "psi": [-180, 180]} # TODO: train
			},
			hover_stabilization_height_in_m = self.hover_stabilization_height_in_m,
			hover_time_in_sec = self.hover_time_in_sec,
			max_diviation_from_hover_height = 0.05,  # TODO: train
			external_drag_disturbance = self.external_drag_disturbance,
			external_wind_disturbance = self.external_wind_disturbance,
			simulation_steps = self.simulation_max_steps,
			level_of_params_uncertanty = self.level_of_params_uncertanty,
			wind_gust_disturbance_time_level = self.wind_gust_disturbance_time_level,
			DR_mode = self.DR_mode
		)
		self.observation_space = Box(
			low = -10000,
			high = 10000,
			shape = (13,),
			dtype = np.float32
		)
		self.action_space = Box(
			low = -1,
			high = 1,
			shape = (4,),
			dtype = np.float32
		)

	def reset(self, seed = None, options = None):
		super().reset(seed = seed)
		init_state_vector, info = self.uav_6dof_env.reset_environment(
			self.np_random
		)
		init_observation_vector = self.get_observation_vector(init_state_vector)
		return init_observation_vector, info

	def step(self, action: np.array):
		action_tensor = torch.from_numpy(action)
		next_state, reward, done = self.uav_6dof_env.step(action_tensor)
		observation = self.get_observation_vector(next_state)
		reward = float(reward)
		terminated = self.uav_6dof_env.episode_termination
		truncated = self.uav_6dof_env.truncated_by_episode_steps
		info = {}

		return (
			observation, reward,
			terminated, truncated,
			info
		)

	def get_observation_vector(self, next_state: dict) -> np.array:
		observation_vector = adf.get_values_as_array(next_state)
		observation_vector = observation_vector.astype(np.float32)
		return observation_vector

if __name__ == "__main__":
	target_hover_height_in_m = 0.8
	env = UAV_6DoF_Adapter_For_GymEnv(
		hover_stabilization_height_in_m = target_hover_height_in_m,
		hover_time_in_sec = 3.0,
		external_drag_disturbance = True,
		external_wind_disturbance = True,
		simulation_max_steps = 1000,
		level_of_params_uncertanty = "delta_1",
		wind_gust_disturbance_time_level = "low",
		DR_mode = True
	)
	check_env(env)
	