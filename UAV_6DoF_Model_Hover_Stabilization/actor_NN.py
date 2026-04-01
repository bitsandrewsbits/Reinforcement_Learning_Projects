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
	def __init__(self, replay_buffer):
		self.replay_buffer = replay_buffer
		self.state = {}

		self.dense_layer_1 = Dense(256, activation = 'relu')
		self.dense_layer_2 = Dense(256, activation = 'relu')
		self.mean_batch = Dense(4)
		self.log_std_batch = Dense(4)

		self.means_tensor = tf.Tensor()
		self.stds_tensor = tf.Tensor()
		self.mean_std_noise_batch = 0

		self.action_t_batch = {}

	def get_action(self, state_t: dict):
		state_t_array = np.array([adf.get_values_as_array(state_t)])
		actions_n_batch = self.get_actions_n_batch(state_t_array)
		action_vector = self.get_action_vector(actions_n_batch)
		return {
			"T_sum": action_vector[0],
			"tau_x": action_vector[1],
			"tau_y": action_vector[2],
			"tau_z": action_vector[3]
		}

	def get_action_vector(self, action_t_batch: tf.Tensor):
		return action_t_batch[0]

	def test_actor_actions_batching(self):
		for _ in range(2):
			batch = self.replay_buffer.get_batch()
			states_n_batch = self.get_states_n_batch(batch)
			actions_n_batch = self.get_actions_n_batch(states_n_batch)
			# print(actions_n_batch)

	def get_actions_n_batch(self, states_n_batch: tf.Tensor):
		self.define_mean_log_std_batches(states_n_batch)
		self.stds_tensor = self.get_std_batch(self.log_stds_tensor)
		normal_noise = tf.random.normal(self.means_tensor.shape)
		self.mean_std_noise_batch = self.means_tensor + self.stds_tensor * normal_noise
		actions_n_batch = tf.tanh(self.mean_std_noise_batch)
		
		return actions_n_batch
		
	def get_mean_batch(self, mean_log_std_batches):
		return mean_log_std_batches[0]

	def get_std_batch(self, log_stds_tensor):
		log_std_batch = tf.clip_by_value(log_stds_tensor, -10, 2)
		return tf.exp(log_std_batch)

	def define_mean_log_std_batches(self, states_n_batch: np.array):
		dense_1 = self.dense_layer_1(states_n_batch)
		dense_2 = self.dense_layer_2(dense_1)
		self.means_tensor = self.mean_batch(dense_2)
		self.log_stds_tensor = self.log_std_batch(dense_2)

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
	buffer = rb.ReplayBuffer(
		pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}),
		10, 1
	)
	buffer.test_buffer_on_random_vectors()
	actor = Actor(buffer)
	actor.test_actor_actions_batching()