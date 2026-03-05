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
		self.action_space = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.F_b_t = np.array([0, 0, self.action_space["T_sum"]])
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.UAV_wing_shoulder_in_m = 0.2
		self.F_g = np.array([0, 0, -self.UAV_mass_in_kg * 9.8])
		self.inertia_moment_J = self.get_moment_inertia_J()
		self.get_force_moment_tau = self.get_force_moment_tau()
		self.state_space_restrictions = state_space_restrictions
		self.hover_stabilization_height = hover_stabilization_height
		self.hover_time_in_sec = hover_time_in_sec

	def get_next_state_vector(self, action_t):
		pass
		# TODO: think, how to create: decomposite each vector to components x,y,z

	def define_next_vector_p(self):
		self.next_state_vector["p"] = self.current_state_vector["p"] + self.current_state_vector["v"] * self.delta_t

	def define_next_vector_v(self):
		F_world = self.get_multiplication_of_quaternion_rotation_matrix_and_F_b_t()
		acceleration = 1 / self.UAV_mass_in_kg * (F_world + self.F_g)
		self.next_state_vector["v"] = self.current_state_vector["v"] + acceleration * self.delta_t

	def get_multiplication_of_quaternion_rotation_matrix_and_F_b_t(self):
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
		multiplication_of_Omega_omega_t_and_q = self.get_multiplication_of_Omega_matrix_and_quaternion()
		self.next_state_vector["q"] = self.current_state_vector["q"] + self.delta_t / 2 * multiplication_of_Omega_omega_t_and_q

	def get_multiplication_of_Omega_matrix_and_quaternion(self):
		Omega_omega_t = np.array([
			[0, -self.current_state_vector["omega"][0], -self.current_state_vector["omega"][1], -self.current_state_vector["omega"][2]],
			[self.current_state_vector["omega"][0], 0, -self.current_state_vector["omega"][2], self.current_state_vector["omega"][1]],
			[self.current_state_vector["omega"][1], self.current_state_vector["omega"][2], 0, -self.current_state_vector["omega"][0]],
			[self.current_state_vector["omega"][2], -self.current_state_vector["omega"][1], self.current_state_vector["omega"][0], 0]
		])
		return Omega_omega_t.dot(self.current_state_vector["q"])

	def define_next_vector_omega(self):
		# TODO: think, how to define moment tau and inertion moment matrix J.
		pass

	def get_moment_inertia_J(self):
		Jx = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jy = 2 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		Jz = 4 * self.UAV_motor_mass_in_kg * self.UAV_wing_shoulder_in_m ** 2
		inertia_moment_J = np.array([
			[Jx, 0, 0], [0, Jy, 0], [0, 0, Jz]
		])
		return inertia_moment_J

	def get_force_moment_tau(self):
		pass

if __name__ == "__main__":
	UAV_model_environment = UAV_6DoF_Model_Environment(
		0.99, {"alpha": 1.8, "beta": 0.5, "gamma": 1.2, "delta": 1.4},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"z": [0, "max_height"], "q1": [0, 0.7], "q2": [0, 0.7]
		},
		0.5, 2
	)