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
	start_position_x: float, start_position_y: float, start_position_z: float,
	hover_stabilization_height_in_m: float, hover_time_in_sec: float,
	external_drag_disturbance: bool, external_wind_disturbance: bool,
	simulation_max_steps: int, train_mode = True):
		super().__init__()

		self.start_position_x = start_position_x
		self.start_position_y = start_position_y
		self.start_position_z = start_position_z
		self.hover_stabilization_height_in_m = hover_stabilization_height_in_m
		self.hover_time_in_sec = hover_time_in_sec
		self.simulation_max_steps = simulation_max_steps
		self.external_drag_disturbance = external_drag_disturbance
		self.external_wind_disturbance = external_wind_disturbance
		self.train_mode = train_mode
		self.uav_6dof_env = uav_env.UAV_6DoF_Model_Environment(
			uav_action_name = 'hover_stabilization',
			discount_gamma = 0.99,
			reward_coefs = {"alpha": 2.0, "beta": 0.4, "gamma": 1.2, "delta": 0.05},
			init_state_vector = {
				"p": np.array([self.start_position_x, self.start_position_y, self.start_position_z]),
				"v": np.array([0, 0, 0]),
				"q": np.array([1, 0, 0, 0]),
				"omega": np.array([0, 0, 0])
			},
			state_space_restrictions = {
				"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
				"angeles": {"fi": [-60, 60], "teta": [-60, 60], "psi": [-180, 180]}
			},
			hover_stabilization_height_in_m = self.hover_stabilization_height_in_m,
			hover_time_in_sec = self.hover_time_in_sec,
			max_diviation_from_hover_height = 0.1,
			external_drag_disturbance = self.external_drag_disturbance,
			external_wind_disturbance = self.external_wind_disturbance,
			simulation_steps = self.simulation_max_steps
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
			self.np_random, train_mode = self.train_mode
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
	env = UAV_6DoF_Adapter_For_GymEnv()
	check_env(env)
	