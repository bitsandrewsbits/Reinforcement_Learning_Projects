# drone 6DoF model environment for UAV hover stabilization
import numpy as np

class UAV_6DoF_Model_Environment:
	def __init__(self, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, state_space_restrictions: dict, 
	hover_stabilization_heigh_in_m, hover_time_in_sec: int):
		self.discount_gamma = discount_gamma
		self.reward_coefs = reward_coefs
		self.delta_t = 0.01
		self.current_state_vector = init_state_vector
		self.next_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		self.action_space_values = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.thrust_force_F_b_t = np.array([0, 0, self.action_space["T_sum"]])
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.UAV_wing_shoulder_in_m = 0.2
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()
		self.get_force_moment_tau = self.get_force_moment_tau()
		self.state_space_restrictions = state_space_restrictions
		self.x_y_coeffs_for_min_max = {"x": 1.1, "y": 1.1}
		self.z_coeff_for_max_height = 1.5
		self.hover_stabilization_height_in_m = hover_stabilization_height
		self.hover_time_in_sec = hover_time_in_sec

		self.early_episode_termination = False

	def start_simulation(self):
		pass

	def get_next_state_vector(self, action_t: dict):
		self.update_action_space_values(action_t)
		self.update_thrust_force_F_b_t()
		self.define_next_vector_p()
		self.define_next_vector_v()
		self.define_next_vector_q()
		self.define_next_vector_omega()

	def update_action_space_values(self, action_t: dict):
		self.action_space_values = action_t

	def restrict_action_t_values(self):
		# TODO: create: check and restrict action_t values in interval [min, max]
		pass

	def check_current_state_vector_components_to_early_episode_termination(self):
		for state_vector_component in ["p", "q"]:
			if self.is_early_episode_termination_by_vector_component(state_vector_component):
				self.early_episode_termination = True
				return 0

		min_T_sum = self.state_space_restrictions["T_sum"][0]
		max_T_sum = self.state_space_restrictions["T_sum"][1]
		if self.current_state_vector["T_sum"] > max_T_sum or self.current_state_vector["T_sum"] < min_T_sum:
			self.early_episode_termination = True

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
		self.next_state_vector["p"] = self.current_state_vector["p"] + self.current_state_vector["v"] * self.delta_t

	def define_next_vector_v(self):
		F_world = self.get_dot_product_of_quaternion_rotation_matrix_and_F_b_t()
		acceleration = 1 / self.UAV_mass_in_kg * (F_world + self.F_g)
		self.next_state_vector["v"] = self.current_state_vector["v"] + acceleration * self.delta_t

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
		return R_q_t.dot(self.F_b_t)

	def define_next_vector_q(self):
		self.quaternion_normalization(self.current_state_vector["q"])
		multiplication_of_Omega_omega_t_and_q = self.get_dot_product_of_Omega_matrix_and_quaternion()
		self.next_state_vector["q"] = self.current_state_vector["q"] + self.delta_t / 2 * multiplication_of_Omega_omega_t_and_q

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
			self.action_space["tau_x"], self.action_space["tau_y"], self.action_space["tau_z"]
		])
		inversed_J = np.linalg.inv(self.inertia_moment_J)
		dot_product_of_J_and_omega = self.inertia_moment_J.dot(self.current_state_vector["omega"])

		self.next_state_vector["omega"] = self.current_state_vector["omega"] + inversed_J.dot(
			moments_tau - np.cross(self.current_state_vector["omega"], dot_product_of_J_and_omega)
		) * self.delta_t

	def get_moment_inertia_J(self):
		Jx = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jy = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jz = 4 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		inertia_moment_J = np.array([
			[Jx, 0, 0], [0, Jy, 0], [0, 0, Jz]
		])
		return inertia_moment_J

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
		0.5, 2
	)