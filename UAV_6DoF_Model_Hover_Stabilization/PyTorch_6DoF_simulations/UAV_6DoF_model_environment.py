# drone 6DoF model environment for UAV hover stabilization
import numpy as np
import pandas as pd
import random
import copy

import domain_randomization as dr

class UAV_6DoF_Model_Environment:
	def __init__(self, uav_action_name: str, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, state_space_restrictions: dict, 
	hover_stabilization_height_in_m, hover_time_in_sec: int,
	max_diviation_from_hover_height, simulation_steps: int,
	external_drag_disturbance = True, external_wind_disturbance = True,
	level_of_params_uncertanty = "delta_0", wind_gust_disturbance_time_level = "low",
	DR_mode = False):
		self.uav_action_name = uav_action_name
		self.discount_gamma = discount_gamma
		self.reward_coefs = reward_coefs
		self.reward_crash_penalty = 100
		self.epsilon = max_diviation_from_hover_height
		self.hover_steps_amount = 0
		self.delta_t_in_sec = 0.01
		self.init_state_vector = init_state_vector
		self.current_state_vector = init_state_vector
		self.current_state_vector_size = sum([val.shape[0] for val in self.current_state_vector.values()])
		self.current_state_euler_angeles = np.array([0.0, 0.0, 0.0])
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		self.next_state_euler_angeles = np.array([0.0, 0.0, 0.0])
		self.battery_mass_in_kg = 0.9
		self.battery_voltage_V = 22.2
		self.UAV_motor_mass_in_kg = 0.0476
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		
		self.UAV_motor_KV = 1300
		self.UAV_max_RPM = self.UAV_motor_KV * self.battery_voltage_V
		self.UAV_RPM_hover = 0.23 * self.UAV_max_RPM
		self.thrust_in_H_for_each_motor = self.UAV_mass_in_kg * 9.8 / 4
		self.omega_hover_in_rad = self.UAV_RPM_hover * (2 * np.pi / 60)
		self.kt = self.thrust_in_H_for_each_motor / self.omega_hover_in_rad ** 2
		self.C_Q_in_Hm_per_RPM2 = 2.44 * 10 ** (-10)
		self.current_step_energy_consumption_E = 0.0
		self.km = self.C_Q_in_Hm_per_RPM2 * (60 / (2 * np.pi)) ** 2
		self.drag_coefficients_baseline = np.array([0.1, 0.1, 0.1])
		self.drag_coefficients = np.array([0.1, 0.1, 0.1])

		self.UAV_wing_shoulder_in_m = 0.1778
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()

		self.x_y_coeffs_for_min_max = {"x": 20, "y": 20}

		self.state_space_restrictions = state_space_restrictions
		self.action_space_restrictions = {
			"w1": [0, 2 * self.omega_hover_in_rad],
			"w2": [0, 2 * self.omega_hover_in_rad],
			"w3": [0, 2 * self.omega_hover_in_rad],
			"w4": [0, 2 * self.omega_hover_in_rad]
		}

		self.hover_stabilization_height_in_m = hover_stabilization_height_in_m
		self.target_p_components_for_hover_stabilization_in_m = [
			self.init_state_vector["p"][0],
			self.init_state_vector["p"][1],
			self.hover_stabilization_height_in_m
		]
		self.hover_action_components = {
			"T_sum": self.UAV_mass_in_kg * 9.8,
			"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0
		}
		self.reward_penalties_by_components = {
			"p": 0.0, "v": 0.0, "angeles": 0.0, "E": 0.0
		}
		if self.uav_action_name == "ascent":
			self.action_space_values = np.array(
				[0, 0, 0, 0], dtype = np.float32
			)
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = np.array([
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad
			], dtype = np.float32)
		self.motors_T = {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0}
		self.T_sum = 0.0
		self.moments_tau = {"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}

		self.thrust_force_F_b_t = np.array([0, 0, self.T_sum])
		self.update_state_space_restrictions()
		self.current_state_vector_reward = 0
		self.hover_time_in_sec = hover_time_in_sec
		self.target_hover_steps_amount = int(self.hover_time_in_sec / self.delta_t_in_sec)

		self.target_done = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.truncated_by_episode_steps = False
		self.current_step = 0
		self.simulation_steps = simulation_steps
		self.RNG = np.random
		self.external_drag_disturbance = external_drag_disturbance
		self.external_wind_disturbance = external_wind_disturbance
		self.level_of_params_uncertanty = level_of_params_uncertanty
		self.wind_gust_disturbance_time_level = wind_gust_disturbance_time_level
		self.domain_randomization = DR_mode
		if self.domain_randomization:
			self.domain_randomizer = dr.Domain_Randomization_For_6DoF(
				drone_mass = self.UAV_mass_in_kg,
				level_of_params_uncertanty = self.level_of_params_uncertanty,
				external_wind_disturbance = self.external_wind_disturbance,
				wind_gust_disturbance_time_level = self.wind_gust_disturbance_time_level,
				episode_steps = self.simulation_steps
			)

	def start_test_simulation(self):
		self.update_state_space_restrictions()
		while self.current_step < self.simulation_steps:
			if self.current_step % 100 == 0:
				print(f"[INFO] Simulation processing {self.current_step // 100} seconds.")
			if self.episode_termination:
				print("[INFO] Simulation(episode) was terminated.")
				self.show_logs()
				return 0
			action = self.get_test_action()
			self.update_action_values(action)
			self.restrict_action_t_values()
			self.next_state_vector, self.current_state_vector_reward, done = self.step(action)
			
		print(f"[INFO] Simulation finished by time = {self.current_step // 100} seconds.")
		self.show_logs()

	def step(self, action_t: np.array):
		self.check_if_episode_is_truncated()
		self.update_action_values(action_t)
		self.update_motors_Ti_for_tau_x_y()
		self.update_T_sum()
		self.update_moments_tau()
		self.define_current_step_energy_consumption_E()
		next_state_vector = self.get_next_state_vector()
		if self.uav_action_name == "ascent":
			current_state_vector_reward = self.get_reward_for_current_state_vector_ascent_stage()
		elif self.uav_action_name == "hover_stabilization":
			current_state_vector_reward = self.get_reward_for_current_state_vector_hover_stabilization_stage()

		self.check_current_state_vector_components_to_episode_termination()
		
		self.make_next_state_vector_as_current()
		self.reset_next_state_vector()
		self.current_step += 1

		return next_state_vector, current_state_vector_reward, self.target_done

	def reset_environment(self, rng = np.random):
		if self.domain_randomization:
			self.init_state_vector = {
				"p": self.domain_randomizer.get_random_state_vector_components(
					np.array([0.0, 0.0, self.hover_stabilization_height_in_m])
				),
				"v": np.array([0, 0, 0]),
				"q": np.array([1, 0, 0, 0]),
				"omega": np.array([0, 0, 0])
			}
			self.UAV_mass_in_kg = self.domain_randomizer.get_random_drone_mass()
			self.drag_coefficients = self.domain_randomizer.get_random_drag(self.drag_coefficients_baseline)

		self.current_state_vector = copy.deepcopy(self.init_state_vector)
		self.current_state_euler_angeles = np.array([0.0, 0.0, 0.0])
		self.next_state_euler_angeles = np.array([0.0, 0.0, 0.0])
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		
		if self.uav_action_name == "ascent":
			self.action_space_values = np.array(
				[0, 0, 0, 0], dtype = np.float32
			)
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = np.array([
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad
			], dtype = np.float32)
		
		self.T_sum = 0.0
		self.thrust_force_F_b_t = np.array([0, 0, self.T_sum])
		self.motors_T = {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0}
		self.moments_tau = {"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.current_state_vector_reward = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.truncated_by_episode_steps = False
		self.current_step = 0
		self.hover_steps_amount = 0
		self.target_done = 0
		self.p_vector_component_episode_logs = {"time_in_sec": [], "x": [], "y": [], "z": []}
		self.RNG = rng
		return self.init_state_vector, {}

	def get_next_state_vector(self):
		self.restrict_action_t_values()
		self.update_thrust_force_F_b_t()
		self.define_next_vector_p()
		self.define_next_vector_v()
		self.define_next_vector_q()
		self.define_next_vector_omega()
		self.next_state_euler_angeles = self.get_euler_angeles(
			self.current_state_vector["q"]
		)

		return self.next_state_vector

	def check_if_episode_is_truncated(self):
		if self.current_step > self.simulation_steps:
			self.truncated_by_episode_steps = True

	def make_next_state_vector_as_current(self):
		self.current_state_vector = {
			"p": self.next_state_vector["p"].copy(),
			"v": self.next_state_vector["v"].copy(),
			"q": self.next_state_vector["q"].copy(),
			"omega": self.next_state_vector["omega"].copy()
		}
		self.next_state_euler_angeles = self.current_state_euler_angeles.copy()

	def reset_next_state_vector(self):
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}

	def update_action_space_values(self, action_t: np.array):
		self.action_space_values = action_t

	def restrict_action_t_values(self):
		for (i, action) in enumerate(self.action_space_restrictions.keys()):
			min_action_value = self.action_space_restrictions[action][0]
			max_action_value = self.action_space_restrictions[action][1]
			if self.action_space_values[i] > max_action_value:
				self.action_space_values[i] = max_action_value
			elif self.action_space_values[i] < min_action_value:
				self.action_space_values[i] = min_action_value

	def check_current_state_vector_components_to_episode_termination(self):
		if self.is_episode_termination_by_p_vector_component():
			self.episode_termination = True
		if self.is_episode_termination_by_euler_angeles_component():
			self.episode_termination = True
		if self.target_done == 1:
			self.episode_termination = True
		return 0

	def is_episode_termination_by_p_vector_component(self):
		for (i, component_value) in enumerate(self.current_state_vector["p"]):
			vector_component_min = list(self.state_space_restrictions["p"].values())[i][0]
			vector_component_max = list(self.state_space_restrictions["p"].values())[i][1]
			if component_value > vector_component_max or component_value < vector_component_min:
				return True
		return False

	def is_episode_termination_by_euler_angeles_component(self):
		restricted_angeles = ["fi", "teta", "psi"]
		for (i, euler_angel) in enumerate(restricted_angeles):
			euler_angel_min = self.state_space_restrictions["angeles"][euler_angel][0]
			euler_angel_max = self.state_space_restrictions["angeles"][euler_angel][1]
			component_value = self.current_state_euler_angeles[i]
			if component_value > euler_angel_max or component_value < euler_angel_min:
				return True
		return False

	def update_state_space_restrictions(self):
		for (p_component, coeff) in self.x_y_coeffs_for_min_max.items():
			self.state_space_restrictions["p"][p_component] = [
				# -coeff * self.hover_stabilization_height_in_m, coeff * self.hover_stabilization_height_in_m
				-5.0, 5.0
			]

		self.state_space_restrictions["p"]["z"] = [0, 5.0]

	def update_thrust_force_F_b_t(self):
		self.thrust_force_F_b_t = np.array([0, 0, self.T_sum])

	def update_action_values(self, action_t: np.array):
		self.action_space_values = self.omega_hover_in_rad * (1 + 0.6 * action_t)

	def update_motors_Ti_for_tau_x_y(self):
		for (action_value, motor_Ti) in zip(self.action_space_values, self.motors_T.keys()):
			self.motors_T[motor_Ti] = self.kt * action_value ** 2

	def update_motors_Ti_for_tau_z(self):
		for (action_value, motor_Ti) in zip(self.action_space_values, self.motors_T.keys()):
			self.motors_T[motor_Ti] = self.km * action_value ** 2

	def update_T_sum(self):
		self.T_sum = sum(self.motors_T.values())

	def define_current_step_energy_consumption_E(self):
		motor_power_consumption = self.C_Q_in_Hm_per_RPM2 * (self.action_space_values ** 3).sum()
		motor_power_consumption = motor_power_consumption.item()
		self.current_step_energy_consumption_E = motor_power_consumption * self.delta_t_in_sec

	def update_moments_tau(self):
		self.moments_tau["tau_x"] = self.UAV_wing_shoulder_in_m * (self.motors_T["T2"] - self.motors_T["T4"])
		self.moments_tau["tau_y"] = self.UAV_wing_shoulder_in_m * (self.motors_T["T3"] - self.motors_T["T1"])
		self.update_motors_Ti_for_tau_z()
		self.moments_tau["tau_z"] = (self.motors_T["T1"] - self.motors_T["T2"] + self.motors_T["T3"] - self.motors_T["T4"])

	def define_next_vector_p(self):
		self.next_state_vector["p"] = self.current_state_vector["p"] + self.current_state_vector["v"] * self.delta_t_in_sec

	def define_next_vector_v(self):
		F_world = self.get_dot_product_of_quaternion_rotation_matrix_and_F_b_t()
		if self.external_drag_disturbance and self.external_wind_disturbance:
			F_drag = self.get_drag_force()
			F_wind = self.domain_randomizer.get_F_wind_gust()
			acceleration = (F_world + self.F_g + F_drag + F_wind) / self.UAV_mass_in_kg
		elif self.external_drag_disturbance:
			F_drag = self.get_drag_force()
			acceleration = (F_world + self.F_g + F_drag) / self.UAV_mass_in_kg
		elif self.external_wind_disturbance:
			F_wind = self.domain_randomizer.get_F_wind_gust()
			acceleration = (F_world + self.F_g + F_wind) / self.UAV_mass_in_kg
		else:
			acceleration = (F_world + self.F_g) / self.UAV_mass_in_kg
		
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

	def get_drag_force(self):
		return -self.drag_coefficients * self.current_state_vector["v"]

	def define_next_vector_q(self):
		multiplication_of_Omega_omega_t_and_q = self.get_dot_product_of_Omega_matrix_and_quaternion()
		self.next_state_vector["q"] = self.current_state_vector["q"] + (self.delta_t_in_sec / 2) * multiplication_of_Omega_omega_t_and_q
		self.normalize_next_state_quaternion()

	def normalize_next_state_quaternion(self):
		quaternion = self.next_state_vector["q"]
		quaternion_module = np.sqrt(quaternion[0] ** 2 + quaternion[1] ** 2 + quaternion[2] ** 2 + quaternion[3] ** 2)
		self.next_state_vector["q"] = quaternion / quaternion_module

	def get_dot_product_of_Omega_matrix_and_quaternion(self):
		Omega_omega_t = np.array([
			[0, -self.current_state_vector["omega"][0], -self.current_state_vector["omega"][1], -self.current_state_vector["omega"][2]],
			[self.current_state_vector["omega"][0], 0, -self.current_state_vector["omega"][2], self.current_state_vector["omega"][1]],
			[self.current_state_vector["omega"][1], self.current_state_vector["omega"][2], 0, -self.current_state_vector["omega"][0]],
			[self.current_state_vector["omega"][2], -self.current_state_vector["omega"][1], self.current_state_vector["omega"][0], 0]
		])
		return Omega_omega_t.dot(self.current_state_vector["q"])

	def get_euler_angeles(self, q: np.array):
		qw, qx, qy, qz = q

		sin_roll = 2.0 * (qw * qx + qy * qz)
		cos_roll = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
		roll_fi = np.arctan2(sin_roll, cos_roll)

		sin_pitch = 2.0 * (qw * qy - qz * qx)
		sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
		pitch_teta = np.arcsin(sin_pitch)

		sin_yaw = 2.0 * (qw * qz + qx * qy)
		cos_yaw = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
		yaw_psi = np.arctan2(sin_yaw, cos_yaw)

		return np.array([roll_fi, pitch_teta, yaw_psi])

	def define_next_vector_omega(self):
		moments_tau = np.array([
			self.moments_tau["tau_x"], self.moments_tau["tau_y"], self.moments_tau["tau_z"]
		])
		inversed_J = np.linalg.inv(self.inertia_moment_J)
		dot_product_of_J_and_omega = self.inertia_moment_J.dot(self.current_state_vector["omega"])

		self.next_state_vector["omega"] = self.current_state_vector["omega"] + inversed_J.dot(
			moments_tau - np.cross(self.current_state_vector["omega"], dot_product_of_J_and_omega)
		) * self.delta_t_in_sec

	def get_moment_inertia_J(self):
		self.Jx = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		self.Jy = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		self.Jz = 4 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		inertia_moment_J = np.array([
			[self.Jx, 0, 0], [0, self.Jy, 0], [0, 0, self.Jz]
		])
		return inertia_moment_J

	def get_reward_for_current_state_vector_ascent_stage(self):
		z_deviation_from_target = self.current_state_vector["p"][2] - self.hover_stabilization_height_in_m

		current_state_vector_reward = 0
		current_state_vector_reward -= 10 * abs(self.current_state_vector["v"][2]) ** 2
		# current_state_vector_reward -= 10 * abs(z_deviation_from_target) ** 2

		if self.current_state_vector["p"][2] <= 0:
		    current_state_vector_reward -= self.reward_crash_penalty

		if abs(z_deviation_from_target) <= self.epsilon:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		return current_state_vector_reward
	
	def get_reward_for_current_state_vector_hover_stabilization_stage(self):
		self.update_reward_penalties_by_vector_components()
		current_state_vector_reward = 0
		sum_penalty = self.get_sum_penalty()
		current_state_vector_reward -= sum_penalty

		if self.hover_steps_amount == self.target_hover_steps_amount:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		if self.current_state_vector["p"][2] <= 0:
			current_state_vector_reward -= self.reward_crash_penalty

		return current_state_vector_reward

	def update_reward_penalties_by_vector_components(self):
		self.reward_penalties_by_components["p"] = self.reward_coefs["alpha"] * self.get_vector_deviation_from_target(
			"p", [0, 1, 2], self.target_p_components_for_hover_stabilization_in_m
		)
		self.reward_penalties_by_components["v"] = self.reward_coefs["beta"] * self.get_vector_deviation_from_target(
			"v", [0, 1, 2], [0.0, 0.0, 0.0]
		)
		self.reward_penalties_by_components["angeles"] = self.reward_coefs["gamma"] * self.get_angeles_deviation_from_target(
			self.current_state_euler_angeles, np.array([0.0, 0.0, 0.0])
		)
		self.reward_penalties_by_components["E"] = self.reward_coefs["delta"] * self.current_step_energy_consumption_E

	def get_sum_penalty(self):
		max_reward_penalty = max(self.reward_penalties_by_components.values())
		sum_penalty = 0
		for (vector_component, value) in self.reward_penalties_by_components.items():
			sum_penalty += value
		return sum_penalty

	def get_vector_deviation_from_target(self, vector: str, components_indexes: list[int], target_vector_values: list):
		vector_deviation_from_target = 0
		for (i, target_vector_value) in zip(components_indexes, target_vector_values):
			vector_deviation_from_target += (
				self.current_state_vector[vector][i] - target_vector_value
			) ** 2
		return np.sqrt(vector_deviation_from_target)

	def get_angeles_deviation_from_target(self, current_euler_angeles: np.array, target_euler_angeles: np.array):
		vector_deviation_from_target = (current_euler_angeles - target_euler_angeles) ** 2
		vector_deviation_from_target = np.sum(vector_deviation_from_target)
		return np.sqrt(vector_deviation_from_target)

	def show_logs(self):
		logs_dataset = pd.DataFrame(self.p_vector_component_episode_logs)
		print(logs_dataset)

	def get_test_action(self):
		return np.array([
				random.randint(-100, 100) / 100,
				random.randint(-100, 100) / 100,
				random.randint(-100, 100) / 100,
				random.randint(-100, 100) / 100,
		])

if __name__ == "__main__":
	UAV_model_environment = UAV_6DoF_Model_Environment(
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
		0.5, 2, 0.2, 10
	)
	UAV_model_environment.start_test_simulation()