# Drone environment(6DoF model) class as Gymnasium API
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
import tensorflow as tf
import numpy as np

import UAV_6DoF_model_environment as uav_env
import additional_functions as adf

class UAV_6DoF_Adapter_For_GymEnv(gym.Env):
	def __init__(self):
		super().__init__()

		self.uav_6dof_env = uav_env.UAV_6DoF_Model_Environment(
			'hover_stabilization', 0.99,
			{"alpha_xy": 0.3, "alpha_z": 5, "beta": 0.5, "gamma": 1.2, "delta": 2},
			{
				"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
				"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
			},
			{
				"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
				"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}
			},
			hover_stabilization_height_in_m = 0.5,
			hover_time_in_sec = 1,
			max_diviation_from_hover_height = 0.2,
			simulation_steps = 1000
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
		init_state_vector, info = self.uav_6dof_env.reset_environment(self.np_random)
		init_observation_vector = self.get_observation_vector(init_state_vector)
		return init_observation_vector, info

	def step(self, action: np.array):
		action_tf = tf.convert_to_tensor(
			action, dtype = tf.float32
		)
		next_state, reward, done = self.uav_6dof_env.step(action_tf)
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
	