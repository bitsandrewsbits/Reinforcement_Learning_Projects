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
	def __init__(self):
		super().__init__()
		self.state = {}

		self.dense_layer_1 = Dense(256, activation = 'relu')
		self.dense_layer_2 = Dense(256, activation = 'relu')
		self.mean_batch = Dense(4)
		self.log_std_batch = Dense(4)

		self.means_tensor = tf.Tensor()
		self.stds_tensor = tf.Tensor()
		self.mean_std_noise_batch = 0

		self.action_t_batch = {}

	def call(self, states_n_batch):
		dense_1 = self.dense_layer_1(states_n_batch)
		dense_2 = self.dense_layer_2(dense_1)
		self.means_tensor = self.mean_batch(dense_2)
		self.log_stds_tensor = self.log_std_batch(dense_2)

		return self.means_tensor, self.log_stds_tensor

	def get_action(self, state_t: dict):
		state_t_array = np.array([adf.get_values_as_array(state_t)])
		self.call(state_t_array)
		actions_n_batch = self.get_actions_n_batch()
		action_vector = self.get_action_vector(actions_n_batch)
		actions_arr = tf.keras.ops.convert_to_numpy(action_vector)
		return {
			"T_sum": actions_arr[0],
			"tau_x": actions_arr[1],
			"tau_y": actions_arr[2],
			"tau_z": actions_arr[3]
		}

	def get_action_vector(self, action_t_batch: tf.Tensor):
		return action_t_batch[0]

	def get_actions_n_batch(self):
		self.stds_tensor = self.get_std_batch(self.log_stds_tensor)
		normal_noise = tf.random.normal(self.means_tensor.shape)
		self.mean_std_noise_batch = self.means_tensor + self.stds_tensor * normal_noise
		actions_n_batch = tf.tanh(self.mean_std_noise_batch)
		# actions_n_batch = tf.tanh(self.means_tensor)
		
		return actions_n_batch
		
	def get_mean_batch(self, mean_log_std_batches):
		return mean_log_std_batches[0]

	def get_std_batch(self, log_stds_tensor):
		log_stds_tensor = tf.clip_by_value(log_stds_tensor, -5, 0)
		std_batch = tf.exp(log_stds_tensor)
		return std_batch

	def get_log_probabilities(self):
		log_probabilities = -0.5 * (
			((self.mean_std_noise_batch - self.means_tensor) / (self.stds_tensor + 1e-8)) ** 2 + \
			2 * self.log_stds_tensor + np.log(2 * np.pi)
		)
		log_probabilities = tf.reduce_sum(log_probabilities, axis = 1, keepdims = True)
		log_probabilities -= tf.reduce_sum(
		    tf.math.log(1 - tf.tanh(self.mean_std_noise_batch) ** 2 + 1e-6),
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