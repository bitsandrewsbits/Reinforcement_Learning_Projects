# drone 6DoF model environment for UAV hover stabilization
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import global_vars as gv

class UAV_6DoF_Model_Environment:
	def __init__(self, uav_action_name: str, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, state_space_restrictions: dict, 
	hover_stabilization_height_in_m, hover_time_in_sec: int,
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
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		self.battery_mass_in_kg = 0.9
		self.battery_voltage_V = 22.2
		self.UAV_motor_mass_in_kg = 0.0476
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		
		self.UAV_motor_KV = 1300
		self.UAV_max_RPM = self.UAV_motor_KV * self.battery_voltage_V
		self.UAV_RPM_hover = 0.18 * self.UAV_max_RPM
		self.thrust_in_H_for_each_motor = self.UAV_mass_in_kg * 9.8 / 4
		self.omega_hover_in_rad = self.UAV_RPM_hover * (2 * np.pi / 60)
		self.kt = self.thrust_in_H_for_each_motor / self.omega_hover_in_rad ** 2

		self.UAV_wing_shoulder_in_m = 0.1778
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()

		self.x_y_coeffs_for_min_max = {"x": 2, "y": 2}

		self.state_space_restrictions = state_space_restrictions
		self.action_space_restrictions = {
			"w1": [0, 2 * self.omega_hover_in_rad],
			"w2": [0, 2 * self.omega_hover_in_rad],
			"w3": [0, 2 * self.omega_hover_in_rad],
			"w4": [0, 2 * self.omega_hover_in_rad]
		}

		self.z_coeff_for_max_height = 1.5
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
		self.reward_penalties_by_vector_components = {
			"x": 0.0, "y": 0.0, "z": 0.0,
			"vx": 0.0, "vy": 0.0, "vz": 0.0,
			"omega_x": 0.0, "omega_y": 0.0, "omega_z": 0.0,
			"q": 0.0
		}
		if self.uav_action_name == "ascent":
			self.action_space_values = tf.constant(
				[0, 0, 0, 0], dtype = tf.float32
			)
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = tf.constant([
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad
			], dtype = tf.float32)
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
		self.current_step = 0
		self.simulation_steps = simulation_steps
		self.p_vector_component_episode_logs = {"time_in_sec": [], "x": [], "y": [], "z": []}
		self.p_vectors_filename_pattern = gv.P_VECTORS_FILENAME_PATTERN

	def start_test_simulation(self):
		self.update_state_space_restrictions()
		while self.current_step < self.simulation_steps:
			if self.current_step % 100 == 0:
				print(f"[INFO] Simulation processing {self.current_step // 100} seconds.")
			if self.episode_termination:
				print("[INFO] Simulation(episode) was terminated.")
				self.show_logs()
				return 0
			self.current_step += 1
			action = self.get_test_action()
			self.update_action_values(action)
			self.restrict_action_t_values()
			print(self.action_space_values)
			self.next_state_vector, self.current_state_vector_reward, done = self.step(action)
			
		print(f"[INFO] Simulation finished by time = {self.current_step // 100} seconds.")
		self.show_logs()

	def step(self, action_t: tf.Tensor):
		self.update_action_values(action_t)
		self.update_motors_Ti()
		self.update_T_sum()
		self.update_moments_tau()
		next_state_vector = self.get_next_state_vector()
		if self.uav_action_name == "ascent":
			current_state_vector_reward = self.get_reward_for_current_state_vector_ascent_stage()
		elif self.uav_action_name == "hover_stabilization":
			current_state_vector_reward = self.get_reward_for_current_state_vector_hover_stabilization_stage()

		self.log_p_vector_components_at_current_step()
		self.check_current_state_vector_components_to_episode_termination()
		
		self.make_next_state_vector_as_current()
		self.reset_next_state_vector()
		self.current_step += 1

		return next_state_vector, current_state_vector_reward, self.target_done

	def reset_environment(self):
		self.current_state_vector = self.init_state_vector
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		if self.uav_action_name == "ascent":
			self.action_space_values = tf.constant(
				[0, 0, 0, 0], dtype = tf.float32
			)
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = tf.constant([
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad,
				self.omega_hover_in_rad
			], dtype = tf.float32)
		self.T_sum = 0.0
		self.thrust_force_F_b_t = np.array([0, 0, self.T_sum])
		self.motors_T = {"T1": 0.0, "T2": 0.0, "T3": 0.0, "T4": 0.0}
		self.moments_tau = {"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.current_state_vector_reward = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.current_step = 0
		self.hover_steps_amount = 0
		self.target_done = 0
		self.p_vector_component_episode_logs = {"time_in_sec": [], "x": [], "y": [], "z": []}

	def get_next_state_vector(self):
		self.restrict_action_t_values()
		self.update_thrust_force_F_b_t()
		self.define_next_vector_p()
		self.define_next_vector_v()
		self.define_next_vector_q()
		self.define_next_vector_omega()

		return self.next_state_vector

	def transition_vector_is_success(self, transition: pd.DataFrame):
		for transition_vector_component_p_index in [0, 1, 2]:
			if self.transition_vector_components_pair_is_success(
			transition, transition_vector_component_p_index):
				continue
			else:
				return False
		return True

	def transition_vector_components_pair_is_success(self, transition: pd.DataFrame, axis_i: int):
		position_p_i = transition["state_t"].values[0][axis_i]
		speed_v_i = transition["state_t"].values[0][axis_i + 3]
		if axis_i == 2: 
			if position_p_i > self.hover_stabilization_height_in_m - self.epsilon:
				if position_p_i < self.hover_stabilization_height_in_m + self.epsilon / 1.2:
					if abs(round(speed_v_i, 4)) < 0.005:
						return True
		else:
			if position_p_i > -self.epsilon / 1.2:
				if position_p_i < self.epsilon / 1.2:
					if abs(round(speed_v_i, 4)) < 0.005:
						return True
		return False

	def make_next_state_vector_as_current(self):
		self.current_state_vector = {
			"p": self.next_state_vector["p"].copy(),
			"v": self.next_state_vector["v"].copy(),
			"q": self.next_state_vector["q"].copy(),
			"omega": self.next_state_vector["omega"].copy()
		}

	def reset_next_state_vector(self):
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}

	def update_action_space_values(self, action_t: tf.Tensor):
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
		if self.is_episode_termination_by_q_vector_component():
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

	def is_episode_termination_by_q_vector_component(self):
		q_restricted_components = [int(component[-1]) for component in self.state_space_restrictions["q"].keys()]
		for q_component in q_restricted_components:
			vector_component_min = self.state_space_restrictions["q"][f"q{q_component}"][0]
			vector_component_max = self.state_space_restrictions["q"][f"q{q_component}"][1]
			component_value = self.current_state_vector["q"][q_component]
			if component_value > vector_component_max or component_value < vector_component_min:
				return True
		return False

	# TODO: change with respect to new action space updates
	def update_state_space_restrictions(self):
		for (p_component, coeff) in self.x_y_coeffs_for_min_max.items():
			self.state_space_restrictions["p"][p_component] = [
				-coeff * self.hover_stabilization_height_in_m, coeff * self.hover_stabilization_height_in_m
			]
		self.state_space_restrictions["p"]["z"] = [-0.2, self.z_coeff_for_max_height * self.hover_stabilization_height_in_m]

	def update_thrust_force_F_b_t(self):
		self.thrust_force_F_b_t = np.array([0, 0, self.T_sum])

	def update_action_values(self, action_t: tf.Tensor):
		self.action_space_values = self.omega_hover_in_rad * (1 + 0.3 * action_t)

	def update_motors_Ti(self):
		for (action_value, motor_Ti) in zip(self.action_space_values, self.motors_T.keys()):
			self.motors_T[motor_Ti] = self.kt * action_value ** 2

	def update_T_sum(self):
		self.T_sum = sum(self.motors_T.values())

	def update_moments_tau(self):
		self.moments_tau["tau_x"] = self.UAV_wing_shoulder_in_m * (self.motors_T["T2"] - self.motors_T["T4"])
		self.moments_tau["tau_y"] = self.UAV_wing_shoulder_in_m * (self.motors_T["T3"] - self.motors_T["T1"])
		self.moments_tau["tay_z"] = self.motors_T["T1"] - self.motors_T["T2"] + self.motors_T["T3"] - self.motors_T["T4"]

	def define_next_vector_p(self):
		self.next_state_vector["p"] = self.current_state_vector["p"] + self.current_state_vector["v"] * self.delta_t_in_sec

	def define_next_vector_v(self):
		F_world = self.get_dot_product_of_quaternion_rotation_matrix_and_F_b_t()
		acceleration = (1 / self.UAV_mass_in_kg) * (F_world + self.F_g)
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
		multiplication_of_Omega_omega_t_and_q = self.get_dot_product_of_Omega_matrix_and_quaternion()
		self.next_state_vector["q"] = self.current_state_vector["q"] + self.delta_t_in_sec / 2 * multiplication_of_Omega_omega_t_and_q
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
		z_deviation_from_target = self.current_state_vector["p"][2] - self.hover_stabilization_height_in_m
		current_state_vector_reward = 0
		if abs(z_deviation_from_target) < self.epsilon:
			self.hover_steps_amount += 1
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["v"][0] + self.next_state_vector["v"][0]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["v"][1] + self.next_state_vector["v"][1]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["v"][2] + self.next_state_vector["v"][2]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["omega"][0] + self.next_state_vector["omega"][0]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["omega"][1] + self.next_state_vector["omega"][1]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - abs(self.current_state_vector["omega"][2] + self.next_state_vector["omega"][2]))
		else:
			self.hover_steps_amount = 0
		sum_penalty = self.get_sum_penalty()
		current_state_vector_reward -= sum_penalty

		if self.hover_steps_amount == self.target_hover_steps_amount:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		return current_state_vector_reward

	def update_reward_penalties_by_vector_components(self):
		self.reward_penalties_by_vector_components["x"] = abs(
			self.current_state_vector["p"][0] - self.target_p_components_for_hover_stabilization_in_m[0]
		)
		self.reward_penalties_by_vector_components["y"] = abs(
			self.current_state_vector["p"][1] - self.target_p_components_for_hover_stabilization_in_m[1]
		)
		self.reward_penalties_by_vector_components["z"] = abs(
			self.current_state_vector["p"][2] - self.hover_stabilization_height_in_m
		)
		self.reward_penalties_by_vector_components["vx"] = abs(self.current_state_vector["v"][0])
		self.reward_penalties_by_vector_components["vy"] = abs(self.current_state_vector["v"][1])
		self.reward_penalties_by_vector_components["vz"] = abs(self.current_state_vector["v"][2])
		self.reward_penalties_by_vector_components["omega_x"] = abs(self.current_state_vector["omega"][0])
		self.reward_penalties_by_vector_components["omega_y"] = abs(self.current_state_vector["omega"][1])
		self.reward_penalties_by_vector_components["omega_z"] = abs(self.current_state_vector["omega"][2])
		self.reward_penalties_by_vector_components["q"] = self.get_vector_deviation_from_target("q", [0, 1, 2, 3], [1, 0, 0, 0])

	def get_sum_penalty(self):
		max_reward_penalty = max(self.reward_penalties_by_vector_components.values())
		sum_penalty = 0
		for (vector_component, value) in self.reward_penalties_by_vector_components.items():
			if max_reward_penalty < 1:
				sum_penalty += value / (max_reward_penalty + 10 ** (-6))
			else:
				sum_penalty += value
		return sum_penalty

	def get_vector_deviation_from_target(self, vector: str, components_indexes: list[int], target_vector_values: list):
		vector_deviation_from_target = 0
		for (i, target_vector_value) in zip(components_indexes, target_vector_values):
			vector_deviation_from_target += (
				self.current_state_vector[vector][i] - target_vector_value
			) ** 2
		return np.sqrt(vector_deviation_from_target)

	def get_vector_by_module(self, vector: str):
		vector_by_module = 0
		for (i, current_vector_component_value) in enumerate(self.current_state_vector[vector]):
			vector_by_module += self.current_state_vector[vector][i] ** 2
		return np.sqrt(vector_by_module)

	def log_p_vector_components_at_current_step(self):
		self.p_vector_component_episode_logs["time_in_sec"].append(self.current_step * self.delta_t_in_sec)
		self.p_vector_component_episode_logs["x"].append(self.current_state_vector["p"][0])
		self.p_vector_component_episode_logs["y"].append(self.current_state_vector["p"][1])
		self.p_vector_component_episode_logs["z"].append(self.current_state_vector["p"][2])

	def save_episode_p_vectors_to_JSONL(self, episode_number: int):
		logs_dataset = pd.DataFrame(self.p_vector_component_episode_logs)
		logs_dataset.to_json(
			f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS["6DoF_model"][self.uav_action_name][1]}/{self.p_vectors_filename_pattern}{episode_number}.jsonl",
			orient = 'records', lines = True
		)

	def show_logs(self):
		logs_dataset = pd.DataFrame(self.p_vector_component_episode_logs)
		print(logs_dataset)

	def get_test_action(self):
		return tf.constant([
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
		0.5, 2, 0.2, 2
	)
	UAV_model_environment.start_test_simulation()