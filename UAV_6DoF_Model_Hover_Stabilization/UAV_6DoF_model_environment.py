# drone 6DoF model environment for UAV hover stabilization
import numpy as np
import time
import pandas as pd
import random

class UAV_6DoF_Model_Environment:
	def __init__(self, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, state_space_restrictions: dict, 
	hover_stabilization_height_in_m, hover_time_in_sec: int,
	simulation_steps: int):
		self.discount_gamma = discount_gamma
		self.reward_coefs = reward_coefs
		self.delta_t_in_sec = 0.01
		self.init_state_vector = init_state_vector
		self.current_state_vector = init_state_vector
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		self.action_space_values = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space_values["T_sum"]])
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.UAV_wing_shoulder_in_m = 0.2
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()
		self.state_space_restrictions = state_space_restrictions
		self.action_space_restrictions = {
			"T_sum": [0, 2 * self.UAV_mass_in_kg * 9.8],
			"tau_x": [-1, 1], "tau_y": [-1, 1], "tau_z": [-1, 1]
		}
		self.x_y_coeffs_for_min_max = {"x": 1.1, "y": 1.1}
		self.z_coeff_for_max_height = 1.25
		self.hover_stabilization_height_in_m = hover_stabilization_height_in_m
		self.target_p_components_for_hover_stabilization_in_m = {
			"x": self.init_state_vector["p"][0] , 
			"y": self.init_state_vector["p"][1],
			"z": self.hover_stabilization_height_in_m
		}
		self.current_state_vector_reward = 0
		self.hover_time_in_sec = hover_time_in_sec

		self.early_episode_termination = False
		self.current_step = 0
		self.simulation_steps = simulation_steps
		self.p_vector_component_logs = {"time_in_sec": [], "x": [], "y": [], "z": []}

	def start_simulation(self):
		self.update_state_space_restrictions()
		while self.current_step < self.simulation_steps:
			if self.current_step % 100 == 0:
				print(f"[INFO] Simulation processing {self.current_step // 100} seconds.")
			if self.early_episode_termination:
				print("[INFO] Simulation(episode) was terminated.")
				return 0
			next_action = self.get_action()
			self.define_next_state_vector(next_action)
			self.make_next_state_vector_as_current()
			self.reset_next_state_vector()
			self.log_p_vector_components_at_current_step()
			
			self.current_step += self.delta_t_in_sec
			time.sleep(self.delta_t_in_sec)
		print(f"[INFO] Simulation finished by time = {self.current_step} seconds.")
		self.show_logs()

	def define_next_state_vector(self, action_t: dict):
		self.update_action_space_values(action_t)
		self.restrict_action_t_values()
		self.update_thrust_force_F_b_t()
		self.define_next_vector_p()
		self.define_next_vector_v()
		self.define_next_vector_q()
		self.define_next_vector_omega()

	def make_next_state_vector_as_current(self):
		self.current_state_vector = self.next_state_vector

	def reset_next_state_vector(self):
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}

	def update_action_space_values(self, action_t: dict):
		self.action_space_values = action_t

	def restrict_action_t_values(self):
		for (action, action_value) in self.action_space_values.items():
			min_action_value = self.action_space_restrictions[action][0]
			max_action_value = self.action_space_restrictions[action][1]
			if action_value > max_action_value:
				self.action_space_values[action] = max_action_value
			elif action_value < min_action_value:
				self.action_space_values[action] = min_action_value

	def check_current_state_vector_components_to_early_episode_termination(self):
		for state_vector_component in ["p", "q"]:
			if self.is_early_episode_termination_by_vector_component(state_vector_component):
				self.early_episode_termination = True
				return 0

	def is_early_episode_termination_by_vector_component(self, vector_component: str):
		for (vector_component, vector_component_min, vector_component_max) in zip(
			self.current_state_vector[vector_component], self.state_space_restrictions[vector_component].values()
		):
			if vector_component > vector_component_max or vector_component < vector_component_min:
				return True
		return False

	def update_state_space_restrictions(self):
		for (p_component, coeff) in self.x_y_coeffs_for_min_max.items():
			self.state_space_restrictions["p"][p_component] = [
				-coeff * self.hover_stabilization_height_in_m, coeff * self.hover_stabilization_height_in_m
			]
		self.state_space_restrictions["p"]["z"] = [0, self.z_coeff_for_max_height * self.hover_stabilization_height_in_m]
		self.state_space_restrictions["T_sum"] = [0, 2 * self.UAV_mass_in_kg * 9.8]

	def update_thrust_force_F_b_t(self):
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space_values["T_sum"]])

	def define_next_vector_p(self):
		self.next_state_vector["p"] = self.current_state_vector["p"] + self.current_state_vector["v"] * self.delta_t_in_sec

	def define_next_vector_v(self):
		F_world = self.get_dot_product_of_quaternion_rotation_matrix_and_F_b_t()
		acceleration = 1 / self.UAV_mass_in_kg * (F_world + self.F_g)
		self.next_state_vector["v"] = self.current_state_vector["v"] + acceleration * self.delta_t_in_sec

	def get_dot_product_of_quaternion_rotation_matrix_and_F_b_t(self):
		R_q_t = np.array([
			[
				1 - 2 * (self.current_state_vector["q"][2] ** 2 + self.current_state_vector["q"][3] ** 2),
				2 * (self.current_state_vector["q"][1] * self.current_state_vector["q"][2] - self.current_state_vector["q"][0] * self.current_state_vector["q"][3]),
				2 * (self.current_state_vector["q"][1] * self.current_state_vector["q"][3] + self.current_state_vector["q"][0] * self.current_state_vector["q"][2])
			],
			[
				2 * (self.current_state_vector["q"][1] * self.current_state_vector["q"][2] + self.current_state_vector["q"][0] * self.current_state_vector["q"][3]),
				1 - 2 * (self.current_state_vector["q"][1] ** 2 + self.current_state_vector["q"][3] ** 2),
				2 * (self.current_state_vector["q"][2] * self.current_state_vector["q"][3] - self.current_state_vector["q"][0] * self.current_state_vector["q"][1])	
			],
			[
				2 * (self.current_state_vector["q"][1] * self.current_state_vector["q"][3] - self.current_state_vector["q"][0] * self.current_state_vector["q"][2]),
				2 * (self.current_state_vector["q"][2] * self.current_state_vector["q"][3] + self.current_state_vector["q"][0] * self.current_state_vector["q"][1]),
				1 - 2 * (self.current_state_vector["q"][1] ** 2 + self.current_state_vector["q"][2] ** 2)
			]
		])
		return R_q_t.dot(self.thrust_force_F_b_t)

	def define_next_vector_q(self):
		self.quaternion_normalization(self.current_state_vector["q"])
		multiplication_of_Omega_omega_t_and_q = self.get_dot_product_of_Omega_matrix_and_quaternion()
		self.next_state_vector["q"] = self.current_state_vector["q"] + self.delta_t_in_sec / 2 * multiplication_of_Omega_omega_t_and_q

	def quaternion_normalization(self, quaternion: np.array):
		quaternion_module = np.sqrt(quaternion[0] ** 2 + quaternion[1] ** 2 + quaternion[2] ** 2 + quaternion[3] ** 2)
		return quaternion / quaternion_module

	def get_dot_product_of_Omega_matrix_and_quaternion(self):
		Omega_omega_t = np.array([
			[0, -self.current_state_vector["omega"][0], -self.current_state_vector["omega"][1], -self.current_state_vector["omega"][2]],
			[self.current_state_vector["omega"][0], 0, -self.current_state_vector["omega"][2], self.current_state_vector["omega"][1]],
			[self.current_state_vector["omega"][1], self.current_state_vector["omega"][2], 0, -self.current_state_vector["omega"][0]],
			[self.current_state_vector["omega"][2], -self.current_state_vector["omega"][1], self.current_state_vector["omega"][0], 0]
		])
		return Omega_omega_t.dot(self.current_state_vector["q"])

	def define_next_vector_omega(self):
		moments_tau = np.array([
			self.action_space_values["tau_x"], self.action_space_values["tau_y"], self.action_space_values["tau_z"]
		])
		inversed_J = np.linalg.inv(self.inertia_moment_J)
		dot_product_of_J_and_omega = self.inertia_moment_J.dot(self.current_state_vector["omega"])

		self.next_state_vector["omega"] = self.current_state_vector["omega"] + inversed_J.dot(
			moments_tau - np.cross(self.current_state_vector["omega"], dot_product_of_J_and_omega)
		) * self.delta_t_in_sec

	def get_moment_inertia_J(self):
		Jx = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jy = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jz = 4 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		inertia_moment_J = np.array([
			[Jx, 0, 0], [0, Jy, 0], [0, 0, Jz]
		])
		return inertia_moment_J

	def get_reward_for_current_state_vector(self):
		p_vector_deviation_from_target = self.get_vector_deviation_by_module("p")
		v_vector_by_module = self.get_vector_by_module("v")
		omega_vector_by_module = self.get_vector_by_module("omega")
		action_vector_by_module = self.get_action_vector_by_module()
		
		current_state_vector_reward = -self.reward_coefs["alpha"] * p_vector_deviation_from_target ** 2 -\
		self.reward_coefs["beta"] * v_vector_by_module ** 2 - self.reward_coefs["gamma"] * omega_vector_by_module ** 2 -\
		self.reward_coefs["delta"] * action_vector_by_module ** 2
		return current_state_vector_reward 

	def get_vector_deviation_by_module(self, vector: str):
		vector_deviation_from_target = 0
		for (i, current_vector_component_value) in enumerate(self.current_state_vector[vector]):
			vector_deviation_from_target += (
				self.current_state_vector[vector][i] - self.target_p_components_for_hover_stabilization_in_m[vector][i]
			) ** 2
		return np.sqrt(vector_deviation_from_target)

	def get_vector_by_module(self, vector: str):
		vector_by_module = 0
		for (i, current_vector_component_value) in enumerate(self.current_state_vector[vector]):
			vector_by_module += self.current_state_vector[vector][i] ** 2
		return np.sqrt(vector_by_module)

	def get_action_vector_by_module(self):
		action_vector_by_module = 0
		for vector_component_value in self.action_space_values.values():
			action_vector_by_module += vector_component_value ** 2
		return np.sqrt(action_vector_by_module)

	def log_p_vector_components_at_current_step(self):
		self.p_vector_component_logs["time_in_sec"] = self.current_step
		self.p_vector_component_logs["x"] = self.current_state_vector["p"][0]
		self.p_vector_component_logs["y"] = self.current_state_vector["p"][1]
		self.p_vector_component_logs["z"] = self.current_state_vector["p"][2]

	def show_logs(self):
		logs_dataset = pd.DataFrame(self.p_vector_component_logs)
		print(logs_dataset)

	def get_action(self):
		return {
			"T_sum": random.randint(0, int(2 * self.UAV_mass_in_kg * 9.8)), 
			"tau_x": random.randint(-1, 1), "tau_y": random.randint(-1, 1), 
			"tau_z": random.randint(-1, 1)
		}

if __name__ == "__main__":
	UAV_model_environment = UAV_6DoF_Model_Environment(
		0.99, {"alpha": 2.8, "beta": 0.5, "gamma": 1.2, "delta": 1.4},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
			"q": {"q1": [0, 0.7], "q2": [0, 0.7]}, "T_sum": [0, "max_T_sum"]
		},
		0.5, 2, 200
	)
	UAV_model_environment.start_simulation()