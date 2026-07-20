# drone 6DoF model environment for UAV hover stabilization
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
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.UAV_wing_shoulder_in_m = 0.2
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()
		self.max_tau_x = self.Jx * 5
		self.max_tau_y = self.Jy * 5
		self.max_tau_z = self.Jz * 5
		self.state_space_restrictions = state_space_restrictions

		self.action_space_restrictions = {
			"T_sum": [0, 4 * self.UAV_mass_in_kg * 9.8],
			"tau_x": [-1, 1], "tau_y": [-1, 1], "tau_z": [-1, 1]
		}
		self.x_y_coeffs_for_min_max = {"x": 2, "y": 2}
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
		if self.uav_action_name == "ascent":
			self.action_space_values = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = {
				"T_sum": self.hover_action_components["T_sum"],
				"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0
			}
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space_values["T_sum"]])
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
			self.next_state_vector, self.current_state_vector_reward = self.step(action)
			
		print(f"[INFO] Simulation finished by time = {self.current_step // 100} seconds.")
		self.show_logs()

	def step(self, action_t: dict):
		next_state_vector = self.get_next_state_vector(action_t)
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
			self.action_space_values = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		elif self.uav_action_name == "hover_stabilization":
			self.action_space_values = {
				"T_sum": self.hover_action_components["T_sum"],
				"tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0
			}
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space_values["T_sum"]])
		self.current_state_vector_reward = 0
		self.episode_termination = False
		self.success_stage_trajectory = False
		self.current_step = 0
		self.hover_steps_amount = 0
		self.target_done = 0
		self.p_vector_component_episode_logs = {"time_in_sec": [], "x": [], "y": [], "z": []}

	def get_next_state_vector(self, action_t: dict):
		self.update_action_space_values(action_t)
		self.restrict_action_t_values()
		self.update_thrust_force_F_b_t()
		self.define_next_vector_p()
		self.define_next_vector_v()
		self.define_next_vector_q()
		self.define_next_vector_omega()

		return self.next_state_vector

	# TODO: think, how to create this method for 3 axises(x, y, z)
	def transition_vector_is_success(self, transition: pd.DataFrame):
		for transition_vector_component_p_index in [0, 1, 2]:
			if self.transition_vector_components_pair_is_success(
			transition, transition_vector_component_p_index):
				continue
			else:
				return False
		return True

	# TODO: create method for one axis, for one (x, vx)-style pair.
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

	def update_state_space_restrictions(self):
		for (p_component, coeff) in self.x_y_coeffs_for_min_max.items():
			self.state_space_restrictions["p"][p_component] = [
				-coeff * self.hover_stabilization_height_in_m, coeff * self.hover_stabilization_height_in_m
			]
		self.state_space_restrictions["p"]["z"] = [-0.2, self.z_coeff_for_max_height * self.hover_stabilization_height_in_m]
		self.state_space_restrictions["T_sum"] = [0.1 * self.hover_action_components["T_sum"], 4 * self.hover_action_components["T_sum"]]

	def update_thrust_force_F_b_t(self):
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space_values["T_sum"]])

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
			self.action_space_values["tau_x"], self.action_space_values["tau_y"], self.action_space_values["tau_z"]
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
		x_deviation_from_target = self.current_state_vector["p"][0] - self.target_p_components_for_hover_stabilization_in_m[0]
		y_deviation_from_target = self.current_state_vector["p"][1] - self.target_p_components_for_hover_stabilization_in_m[1]
		z_deviation_from_target = self.current_state_vector["p"][2] - self.hover_stabilization_height_in_m
		# omega_deviation_from_target = self.get_vector_by_module("omega")
		quaternion_deviation_from_target = self.get_vector_deviation_from_target("q", [0, 1, 2, 3], [1, 0, 0, 0])

		current_state_vector_reward = 0
		if abs(z_deviation_from_target) < self.epsilon:
			self.hover_steps_amount += 1
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 45 * abs(self.current_state_vector["v"][0] + self.next_state_vector["v"][0]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 45 * abs(self.current_state_vector["v"][1] + self.next_state_vector["v"][1]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 45 * abs(self.current_state_vector["v"][2] + self.next_state_vector["v"][2]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 50 * abs(self.current_state_vector["omega"][0] + self.next_state_vector["omega"][0]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 50 * abs(self.current_state_vector["omega"][1] + self.next_state_vector["omega"][1]))
			current_state_vector_reward += self.hover_extra_reward * (0.1 - 50 * abs(self.current_state_vector["omega"][2] + self.next_state_vector["omega"][2]))
		else:
			self.hover_steps_amount = 0

		current_state_vector_reward -= 99 * abs(x_deviation_from_target)
		current_state_vector_reward -= 99 * abs(y_deviation_from_target)
		current_state_vector_reward -= 99 * abs(z_deviation_from_target)
		current_state_vector_reward -= 110 * abs(self.current_state_vector["v"][0])
		current_state_vector_reward -= 110 * abs(self.current_state_vector["v"][1])
		current_state_vector_reward -= 110 * abs(self.current_state_vector["v"][2])
		current_state_vector_reward -= 110 * abs(self.current_state_vector["omega"][0])
		current_state_vector_reward -= 110 * abs(self.current_state_vector["omega"][1])
		current_state_vector_reward -= 110 * abs(self.current_state_vector["omega"][2])
		current_state_vector_reward -= 110 * quaternion_deviation_from_target

		if self.hover_steps_amount == self.target_hover_steps_amount:
			current_state_vector_reward += 500
			self.target_done = 1
			self.success_stage_trajectory = True
		else:
			self.success_stage_trajectory = False

		return current_state_vector_reward

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

	def get_action_deviation_by_module(self):
		action_deviation_by_module = 0
		for action_component in self.action_space_values:
			action_deviation_by_module += (
				self.action_space_values[action_component] - self.hover_action_components[action_component]
			) ** 2
		return np.sqrt(action_deviation_by_module)

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
		return {
			"T_sum": self.UAV_mass_in_kg * 9.8 + random.uniform(-1, 1), 
			"tau_x": random.randint(-1, 1) / 20, "tau_y": random.randint(-1, 1) / 20,
			"tau_z": random.randint(-1, 1) / 50
		}

if __name__ == "__main__":
	UAV_model_environment = UAV_6DoF_Model_Environment(
		0.99, {"alpha_xy": 0.3, "alpha_z": 5, "beta": 0.5, "gamma": 1.2, "delta": 2},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
			"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
		},
		0.5, 2, 1000
	)
	UAV_model_environment.start_test_simulation()