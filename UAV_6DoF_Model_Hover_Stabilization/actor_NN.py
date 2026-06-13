# Actor NN for SAC
# state_t -> NN -> action_t
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import replay_buffer as rb
import UAV_6DoF_model_environment as uav_env
import additional_functions as adf

class Actor(tf.keras.Model):
	def __init__(self, uav_env = "6DoF_model"):
		super().__init__()
		self.uav_env = uav_env

		self.dense_layer_1 = Dense(256, activation = 'relu')
		self.dense_layer_2 = Dense(256, activation = 'relu')
		
		if self.uav_env == "6DoF_model":
			output_values_amount = 4
		elif self.uav_env == "1D_model":
			output_values_amount = 1
		
		self.mean_batch = Dense(output_values_amount)
		self.log_std_batch = Dense(output_values_amount)

		self.action_t_batch = {}

	def call(self, states_n_batch):
		dense_1 = self.dense_layer_1(states_n_batch)
		dense_2 = self.dense_layer_2(dense_1)
		means_tensor = self.mean_batch(dense_2)
		log_stds_tensor = self.log_std_batch(dense_2)

		mean_std_noise_batch = self.get_mean_std_noise_batch(means_tensor, log_stds_tensor)
		actions_n_batch = self.get_actions_n_batch(mean_std_noise_batch)
		log_probabilities = self.get_log_probabilities(mean_std_noise_batch, means_tensor, log_stds_tensor)
		# print("Action:", actions_n_batch)
		# print("Log probs:", log_probabilities)
		return actions_n_batch, log_probabilities

	def get_action(self, state_t: dict):
		self.call(state_t_array)

	def get_action_vector(self, action_t_batch: tf.Tensor):
		return action_t_batch[0]

	def get_actions_n_batch(self, mean_std_noise_batch):
		actions_n_batch = tf.tanh(mean_std_noise_batch)
		return actions_n_batch

	def get_mean_std_noise_batch(self, means_batch, log_stds_batch):
		stds_batch = self.get_std_batch(log_stds_batch)
		normal_noise = tf.random.normal(means_batch.shape)
		mean_std_noise_batch = means_batch + stds_batch * normal_noise
		return mean_std_noise_batch
		
	def get_mean_batch(self, mean_log_std_batches):
		return mean_log_std_batches[0]

	def get_std_batch(self, log_stds_tensor):
		log_stds_tensor = tf.clip_by_value(log_stds_tensor, -10, 2)
		std_batch = tf.exp(log_stds_tensor)
		return std_batch

	def get_log_probabilities(self, mean_std_noise_batch, means_batch, log_stds_batch):
		stds_batch = self.get_std_batch(log_stds_batch)
		mean_std_noise_batch = self.get_mean_std_noise_batch(means_batch, log_stds_batch)
		log_probabilities = -0.5 * (
			((mean_std_noise_batch - means_batch) / (stds_batch + 1e-8)) ** 2 + \
			2 * log_stds_batch + np.log(2 * np.pi)
		)
		log_probabilities = tf.reduce_sum(log_probabilities, axis = 1, keepdims = True)
		log_probabilities -= tf.reduce_sum(
		    tf.math.log(1 - tf.tanh(mean_std_noise_batch) ** 2 + 1e-6),
		    axis = 1,
		    keepdims = True
		)
		return log_probabilities

	def get_states_n_batch(self, batch: pd.DataFrame):
		return np.array(list(batch["state_t"].values))

if __name__ == "__main__":
	buffer = rb.ReplayBuffer(100, 32)
	buffer.test_buffer_on_random_vectors()
	test_batch = buffer.get_batch()
	states_n_batch = buffer.get_states_actions_rewards_dones_next_states(test_batch)[0]
	actor = Actor()
	pred = actor(states_n_batch)
	print(pred)