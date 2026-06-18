# drone 6DoF model environment for UAV hover stabilization
import numpy as np
import pandas as pd
import random
import global_vars as gv
import copy

class UAV_1D_Model_Environment:
	def __init__(self, uav_action_name: str, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, hover_stabilization_height_in_m, hover_time_in_sec: int,
	max_diviation_from_hover_height,
	simulation_steps: int):
		self.uav_action_name = uav_action_name
		self.discount_gamma = discount_gamma
		self.reward_coefs = reward_coefs
		self.hover_extra_reward = 5
		self.reward_crash_penalty = 20
		self.epsilon = max_diviation_from_hover_height
		self.hover_steps_amount = 0
		self.delta_t_in_sec = 0.01
		self.init_state_vector = init_state_vector
		self.current_state_vector = init_state_vector
		self.current_state_vector_size = sum([val.shape[0] for val in self.current_state_vector.values()])
		self.next_state_vector = {"z": [], "vz": []}
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.UAV_wing_shoulder_in_m = 0.2
		self.F_g = np.array([-self.UAV_mass_in_kg * 9.8])
		self.k_vz = 0.1
		self.state_space_restrictions = {"z": 0}
		self.action_space_restrictions = {
			"T_sum": [0, 4 * self.UAV_mass_in_kg * 9.8]
		}
		self.z_coeff_for_max_height = 1.5
		self.hover_stabilization_height_in_m = hover_stabilization_height_in_m
		self.target_p_components_for_hover_stabilization_in_m = [
			self.hover_stabilization_height_in_m
		]
		self.hover_action_components = {
			"T_sum": self.UAV_mass_in_kg * 9.8
		}
		if self.uav_action_name == "ascent":
			self.action_space_values = {"T_sum": 0.0}
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = {
				"T_sum": self.hover_action_components["T_sum"]
			}
		self.update_state_space_restrictions()
		self.current_state_vector_reward = 0
		self.hover_time_in_sec = hover_time_in_sec
		self.target_hover_steps_amount = int(self.hover_time_in_sec / self.delta_t_in_sec)

		self.target_done = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.evaluation_simulation = False
		self.current_step = 0
		self.simulation_steps = simulation_steps
		self.z_height_episode_logs = {"time_in_sec": [], "z": []}
		self.p_vectors_filename_pattern = gv.P_VECTORS_FILENAME_PATTERN

	def start_test_simulation(self):
		self.update_state_space_restrictions()
		while self.current_step < self.simulation_steps:
			if self.current_step % 100 == 0:
				print(f"[INFO] Simulation processing {self.current_step // 100} seconds.")
			if self.early_episode_termination:
				print("[INFO] Simulation(episode) was terminated.")
				self.show_logs()
				return 0
			self.current_step += 1
			action = self.get_test_action()
			self.next_state_vector, self.current_state_vector_reward, done = self.step(action)
			
		print(f"[INFO] Simulation finished by time = {self.current_step // 100} seconds.")
		self.show_logs()

	def step(self, action_t: dict):
		next_state_vector = self.get_next_state_vector(action_t)
		if self.uav_action_name == "ascent":
			current_state_vector_reward = self.get_reward_for_current_state_vector_ascent_stage()
		elif self.uav_action_name == "hover_stabilization":
			current_state_vector_reward = self.get_reward_for_current_state_vector_hover_stabilization_stage()

		self.log_z_height_current_step()
		self.check_current_state_vector_components_to_episode_termination()
		
		self.make_next_state_vector_as_current()
		self.reset_next_state_vector()
		self.current_step += 1

		return next_state_vector, current_state_vector_reward, self.target_done

	def reset_environment(self):
		add_deviation_for_z = 0
		add_deviation_for_Tsum = 0
		if self.evaluation_simulation:
			add_deviation_for_z = random.randint(-2, 20) / 1000
			add_deviation_for_Tsum = random.randint(-3, 3) / 1000

		self.current_state_vector = copy.deepcopy(self.init_state_vector)
		self.current_state_vector["z"][0] += add_deviation_for_z
		self.next_state_vector = {"z": [], "vz": []}
		if self.uav_action_name == "ascent":
			self.action_space_values = {"T_sum": 0.0}
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = {
				"T_sum": self.hover_action_components["T_sum"] + add_deviation_for_Tsum
			}
		self.current_state_vector_reward = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.current_step = 0
		self.hover_steps_amount = 0
		self.target_done = 0
		self.z_height_episode_logs = {"time_in_sec": [], "z": []}

	def get_next_state_vector(self, action_t: dict):
		self.update_action_space_values(action_t)
		self.restrict_action_t_values()
		self.define_next_z()
		self.define_next_vz()

		return self.next_state_vector

	def transition_vector_is_success(self, transition: pd.DataFrame):
		height_z = transition["state_tp1"].values[0][0]
		speed_vz = transition["state_tp1"].values[0][1]
		if height_z > self.hover_stabilization_height_in_m - self.epsilon / 1.2:
			if height_z < self.hover_stabilization_height_in_m + self.epsilon / 1.2:
				if abs(round(speed_vz, 4)) < 0.005:
					return True
		return False

	def make_next_state_vector_as_current(self):
		self.current_state_vector = {
			"z": self.next_state_vector["z"].copy(),
			"vz": self.next_state_vector["vz"].copy(),
		}

	def reset_next_state_vector(self):
		self.next_state_vector = {"z": [], "vz": []}

	def update_action_space_values(self, action_t: dict):
		self.action_space_values = action_t

	def restrict_action_t_values(self):
		min_action_value = self.action_space_restrictions["T_sum"][0]
		max_action_value = self.action_space_restrictions["T_sum"][1]
		if self.action_space_values["T_sum"] > max_action_value:
			self.action_space_values["T_sum"] = max_action_value
		elif self.action_space_values["T_sum"] < min_action_value:
			self.action_space_values["T_sum"] = min_action_value

	def check_current_state_vector_components_to_episode_termination(self):
		if self.is_early_episode_termination_by_z():
			self.episode_termination = True
		elif self.target_done == 1:
			self.episode_termination = True
		elif self.current_step > self.simulation_steps:
			self.episode_termination = True
		return 0

	def is_early_episode_termination_by_z(self):
		vector_component_min = self.state_space_restrictions["z"][0]
		vector_component_max = self.state_space_restrictions["z"][1]
		if self.current_state_vector["z"][0] > vector_component_max or self.current_state_vector["z"][0] < vector_component_min:
			return True
		return False

	def update_state_space_restrictions(self):
		self.state_space_restrictions["z"] = [-0.2, self.z_coeff_for_max_height * self.hover_stabilization_height_in_m]
		self.action_space_restrictions["T_sum"] = [0.1 * self.hover_action_components["T_sum"], 4 * self.hover_action_components["T_sum"]]

	def define_next_z(self):
		self.next_state_vector["z"] = self.current_state_vector["z"] + self.current_state_vector["vz"] * self.delta_t_in_sec

	def define_next_vz(self):
		F_world = self.action_space_values["T_sum"]
		acceleration = (1 / self.UAV_mass_in_kg) * (F_world + self.F_g)
		self.next_state_vector["vz"] = self.current_state_vector["vz"] + acceleration * self.delta_t_in_sec

	def get_reward_for_current_state_vector_ascent_stage(self):
		z_deviation_from_target = self.current_state_vector["z"][0] - self.hover_stabilization_height_in_m

		current_state_vector_reward = 0
		current_state_vector_reward -= 10 * abs(self.current_state_vector["vz"][0]) ** 2
		# current_state_vector_reward -= 10 * abs(z_deviation_from_target) ** 2

		if self.current_state_vector["z"][0] <= 0:
		    current_state_vector_reward -= self.reward_crash_penalty

		if abs(z_deviation_from_target) <= self.epsilon:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		return current_state_vector_reward
	
	def get_reward_for_current_state_vector_hover_stabilization_stage(self):
		z_deviation_from_target = self.current_state_vector["z"][0] - self.hover_stabilization_height_in_m
		vz_deviation_from_target = self.get_vector_by_module("vz")

		current_state_vector_reward = 0
		if abs(z_deviation_from_target) < self.epsilon:
			self.hover_steps_amount += 1
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 20 * abs(self.current_state_vector["vz"][0] + self.next_state_vector["vz"][0]))
		else:
			self.hover_steps_amount = 0
		
		# current_state_vector_reward += 50 * (self.hover_steps_amount / self.target_hover_steps_amount)
		current_state_vector_reward -= 80 * abs(z_deviation_from_target)
		current_state_vector_reward -= 100 * abs(self.current_state_vector["vz"][0])

		if self.hover_steps_amount == self.target_hover_steps_amount:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		return current_state_vector_reward

	def get_vector_by_module(self, vector: str):
		vector_by_module = 0
		for (i, current_vector_component_value) in enumerate(self.current_state_vector[vector]):
			vector_by_module += self.current_state_vector[vector][i] ** 2
		return np.sqrt(vector_by_module)

	def get_action_deviation_by_module(self):
		action_deviation_by_module = 0
		for action_component in self.action_space_values:
			action_deviation_by_module += (
				self.action_space_values[action_component] - self.hover_action_components[action_component]
			) ** 2
		return np.sqrt(action_deviation_by_module)

	def log_z_height_current_step(self):
		self.z_height_episode_logs["time_in_sec"].append(self.current_step * self.delta_t_in_sec)
		self.z_height_episode_logs["z"].append(self.current_state_vector["z"][0])

	def save_episode_p_vectors_to_JSONL(self, episode_number: int):
		logs_dataset = pd.DataFrame(self.z_height_episode_logs)
		logs_dataset.to_json(
			f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][1]}/{self.p_vectors_filename_pattern}{episode_number}.jsonl",
			orient = 'records', lines = True
		)

	def show_logs(self):
		logs_dataset = pd.DataFrame(self.z_height_episode_logs)
		print(logs_dataset)

	def get_test_action(self):
		return {
			"T_sum": self.UAV_mass_in_kg * 9.8 + random.uniform(-1, 1)
		}

if __name__ == "__main__":
	UAV_model_environment = UAV_1D_Model_Environment(
		"hover_stabilization", 0.99, {"alpha_z": 5, "beta": 0.5, "delta": 2},
		{
			"z": np.array([0]), "vz": np.array([0])
		},
		0.5, 2, 0.2, 1000
	)
	UAV_model_environment.start_test_simulation()