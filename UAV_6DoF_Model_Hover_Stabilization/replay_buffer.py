# Replay Buffer for transition vectors saving per episode
# transition vector structure - (s_t, a_t, r_t, s_t+1)
# replay buffer structure = [(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2),..., (s_T-1, a_T-1, r_T-1, s_T)]
# trajectory -> DataFrame per episode
# trajectories = [trajectory-df-1, trajectory-df-2, ... , trajectory-df-N]

from random import randint
import pandas as pd
import numpy as np
import additional_functions as adf
import tensorflow as tf

class ReplayBuffer:
	def __init__(self, buffer_max_transition_vectors: int, batch_size: int):
		self.buffer_max_transition_vectors = buffer_max_transition_vectors
		self.buffer = pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []})
		self.batch_size = batch_size

	def test_buffer_on_random_vectors(self):
		for i in range(self.buffer_max_transition_vectors):
			transition_vector = self.get_transition_vector(
				self.generate_state_t(), self.generate_action_t(),
				self.generate_reward_t(), self.generate_done(),
				self.generate_state_tp1()
			)
			self.add_transition_vector_to_buffer(transition_vector, i)

	def test_buffer_batch_getting(self):
		for _ in range(1):
			batch = self.get_batch()
			print("Current batch:")
			print(batch)
			print("Separated vectors:")
			states, actions, rewards, dones, next_states = self.get_states_actions_rewards_dones_next_states(batch)
			print("states:", states)
			print("actions:", actions)
			print("rewards:", rewards)
			print("dones:", dones)
			print("next_states:", next_states)

	def add_transition_vector_to_buffer(self, transition_vector: pd.DataFrame):
		if self.buffer.shape[0] < self.buffer_max_transition_vectors:
			self.buffer = pd.concat([self.buffer, transition_vector], ignore_index = True)
		else:
			print("[WARN] Reach max size of buffer!")

	def update_buffer_with_new_transition_vector(self, transition_vector: pd.DataFrame, passed_steps: int):
		if passed_steps < self.buffer_max_transition_vectors:
			self.buffer = pd.concat([self.buffer, transition_vector], ignore_index = True)
		else:
			updated_index = passed_steps % self.buffer_max_transition_vectors
			self.buffer.loc[updated_index] = transition_vector.loc[0]

	def update_success_buffer_with_new_transition(self, transition_vector: pd.DataFrame):
		self.buffer = pd.concat([self.buffer, transition_vector], ignore_index = True)
		if self.buffer.shape[0] + transition_vector.shape[0] > self.buffer_max_transition_vectors:
			self.buffer = self.buffer[self.buffer.shape[0] - self.buffer_max_transition_vectors:]

	def get_transition_vector(self, state_t: tf.Tensor, action_t: dict, reward_t: float, done: int, state_tp1: dict):
		transition_vector = {"state_t": [], "action_t": [], "reward_t": [], "target_done": [], "state_tp1": []}
		transition_vector["state_t"].append(tf.keras.ops.convert_to_numpy(tf.squeeze(state_t)))
		transition_vector["action_t"].append(list(action_t.values()))
		transition_vector["reward_t"].append(reward_t)
		transition_vector["target_done"].append(done)
		transition_vector["state_tp1"].append(adf.get_values_as_array(state_tp1))

		return pd.DataFrame(transition_vector)

	def get_batch(self, success_buffer = None) -> pd.DataFrame:
		if success_buffer is not None and success_buffer.buffer.shape[0] >= success_buffer.batch_size:
			result_batch_size = self.batch_size + success_buffer.batch_size
			train_transitions_batch = self.buffer.sample(self.batch_size, axis = 0)
			success_transitions_batch = success_buffer.buffer.sample(success_buffer.batch_size, axis = 0)
			result_batch = pd.concat([train_transitions_batch, success_transitions_batch], ignore_index = True)
			return result_batch.sample(result_batch_size, axis = 0)
		else:
			return self.buffer.sample(self.batch_size, axis = 0)

	def get_states_actions_rewards_dones_next_states(self, batch: pd.DataFrame):
		states = tf.convert_to_tensor([state for state in batch["state_t"].values])
		actions = [action for action in batch["action_t"].values]
		rewards = batch["reward_t"].values
		dones = batch["target_done"].values
		next_states = tf.convert_to_tensor([next_state for next_state in batch["state_tp1"].values])
		return states, actions, rewards, dones, next_states

	def generate_state_t(self):
		state_t = {
			"p": [randint(-10, 10), randint(-10, 10), randint(0, 10)],
			"v": [randint(0, 20), randint(0, 20), randint(0, 20)],
			"q": [randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)],
			"omega": [randint(0, 5), randint(0, 5), randint(0, 5)]
		}
		return state_t

	def generate_action_t(self):
		return {
			"T_sum": randint(0, 80), "tau_x": randint(-3, 3),
			"tau_y": randint(-3, 3), "tau_z": randint(-3, 3)
		}

	def generate_reward_t(self):
		return randint(-80, 80)

	def generate_done(self):
		return randint(0, 1)

	def generate_state_tp1(self):
		state_tp1 = {
			"p": [randint(-10, 10), randint(-10, 10), randint(0, 10)],
			"v": [randint(0, 20), randint(0, 20), randint(0, 20)],
			"q": [randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)],
			"omega": [randint(0, 5), randint(0, 5), randint(0, 5)]
		}
		return state_tp1

if __name__ == "__main__":
	buffer = ReplayBuffer(25, 3)
	buffer.test_buffer_on_random_vectors()
	buffer.test_buffer_batch_getting()