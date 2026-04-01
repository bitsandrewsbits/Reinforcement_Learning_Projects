# critic NNs
import additional_functions as adf
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
	def __init__(self):
		super().__init__()

		self.Q1_dense_layer_1 = Dense(256, activation = 'relu')
		self.Q1_dense_layer_2 = Dense(256, activation = 'relu')
		self.Q1_s_a_output = Dense(1)

		self.Q2_dense_layer_1 = Dense(256, activation = 'relu')
		self.Q2_dense_layer_2 = Dense(256, activation = 'relu')
		self.Q2_s_a_output = Dense(1)

	def call(self, states_actions_batch):
		Q1_dense_l1 = self.Q1_dense_layer_1(states_actions_batch)
		Q1_dense_l2 = self.Q1_dense_layer_2(Q1_dense_l1)
		Q1_s_a_value = self.Q1_s_a_output(Q1_dense_l2)

		Q2_dense_l1 = self.Q2_dense_layer_1(states_actions_batch)
		Q2_dense_l2 = self.Q2_dense_layer_2(Q2_dense_l1)
		Q2_s_a_value = self.Q2_s_a_output(Q2_dense_l2)
		
		return Q1_s_a_value, Q2_s_a_value

	def get_states_actions_batch(self, states_n_batch, actions_n_batch):
		return tf.concat([states_n_batch, actions_n_batch], axis = 1)

if __name__ == "__main__":
	test_state_t = {
		"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
		"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
	}
	test_action_t = {"T_sum": 4, "tau_x": 1, "tau_y": 5, "tau_z": 6}
	test_critic = Critic()
