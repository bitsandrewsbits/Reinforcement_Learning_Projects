# critic NNs
import additional_functions as adf
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
	def __init__(self):
		self.Q1_dense_layer_1 = Dense(256, activation = 'relu')
		self.Q1_dense_layer_2 = Dense(256, activation = 'relu')
		self.Q1_s_a_output_value = Dense(1)

		self.Q2_dense_layer_1 = Dense(256, activation = 'relu')
		self.Q2_dense_layer_2 = Dense(256, activation = 'relu')
		self.Q2_s_a_output_value = Dense(1)

		self.states_actions_batch = tf.Tensor()
		self.Q1_s_a_batch = tf.Tensor()
		self.Q2_s_a_batch = tf.Tensor()
		self.min_Q1_Q2_batch = tf.Tensor()

	def get_critic_loss(self, states_n_batch, actions_n_batch, target_Q):
		self.states_actions_batch = self.get_states_actions_batch(states_n_batch, actions_n_batch)
		self.define_min_Q1_Q2_batch(self.states_actions_batch)
		critic_loss = (self.Q1_s_a_batch - target_Q) ** 2 + (self.Q2_s_a_batch - target_Q) ** 2
		return critic_loss

	def define_min_Q1_Q2_batch(self, states_actions_batch):
		self.Q1_s_a_batch = self.get_Q1_a_s_batch(states_actions_batch)
		self.Q2_s_a_batch = self.get_Q2_a_s_batch(states_actions_batch)
		self.min_Q1_Q2_batch = min(self.Q1_s_a_batch, self.Q2_s_a_batch)

	def get_Q1_a_s_batch(self, states_actions_batch):
		dense_layer_1 = self.Q1_dense_layer_1(states_actions_batch)
		dense_layer_2 = self.Q1_dense_layer_2(dense_layer_1)
		Q1_s_a_value = self.Q1_s_a_output_value(dense_layer_2)
		return Q1_s_a_value

	def get_Q2_a_s_batch(self, states_actions_batch):
		dense_layer_1 = self.Q2_dense_layer_1(states_actions_batch)
		dense_layer_2 = self.Q2_dense_layer_2(dense_layer_1)
		Q2_s_a_value = self.Q2_s_a_output_value(dense_layer_2)
		return Q2_s_a_value

	def get_states_actions_batch(self, states_n_batch, actions_n_batch):
		state_action_batch = tf.concat([states_n_batch, actions_n_batch], axis = 1)
		return state_action_batch

	def get_target_Q(self, rewards, gamma, alpha, next_actions):
		min_Q1_Q2_batch = self.get_min_Q1_Q2_batch()
		target_Q = rewards + gamma * (min_Q1_Q2_batch - alpha * tf.math.log(next_actions))
		return target_Q

if __name__ == "__main__":
	test_state_t = {
		"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
		"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
	}
	test_action_t = {"T_sum": 4, "tau_x": 1, "tau_y": 5, "tau_z": 6}
	test_critic = Critic()
	test_critic.get_min_Q1_Q2_batch(test_state_t, test_action_t)
