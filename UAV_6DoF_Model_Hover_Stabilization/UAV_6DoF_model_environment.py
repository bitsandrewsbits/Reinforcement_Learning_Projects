# drone 6DoF model environment for UAV hover stabilization
import numpy as np

class UAV_6DoF_Model_Environment:
	def __init__(self, discount_gamma, reward_coefs: dict, 
	init_state_vector: dict, state_space_restrictions: dict, 
	hover_stabilization_heigh_in_m, hover_time_in_sec: int):
		self.discount_gamma = discount_gamma
		self.reward_coefs = reward_coefs
		self.init_state_vector = init_state_vector
		self.current_state_vector = {"p": [], "v": [], "q": [], "omega": []}
		self.action_space = {"T_sum": 0.0, "tau_x": 0.0, "tau_y": 0.0, "tau_z": 0.0}
		self.state_space_restrictions = state_space_restrictions
		self.hover_stabilization_height = hover_stabilization_height
		self.hover_time_in_sec = hover_time_in_sec

	def get_next_state_vector(self, action_t):
		pass
		# TODO: think, how to create: decomposite each vector to components x,y,z


if __name__ == "__main__":
	UAV_model_environment = UAV_6DoF_Model_Environment(
		0.99, {"alpha": 1.8, "beta": 0.5, "gamma": 1.2, "delta": 1.4},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([0, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"z": [0, "max_height"], "q1": [0, 0.7], "q2": [0, 0.7]
		},
		0.5, 2
	)