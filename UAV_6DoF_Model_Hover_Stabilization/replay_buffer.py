# Replay Buffer for transition vectors saving per episode
# transition vector structure - (s_t, a_t, r_t, s_t+1)
# replay buffer structure = [(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2),..., (s_T-1, a_T-1, r_T-1, s_T)]
# trajectory -> DataFrame per episode
# trajectories = [trajectory-df-1, trajectory-df-2, ... , trajectory-df-N]

from random import randint
import pandas as pd

class ReplayBuffer:
	def __init__(self, buffer: pd.DataFrame, buffer_max_transition_vectors: int, batch_size: int):
		self.buffer_max_transition_vectors = buffer_max_transition_vectors
		self.buffer = buffer
		self.batch_size = batch_size

	def test_buffer_on_random_vectors(self):
		for _ in range(self.buffer_max_transition_vectors):
			transition_vector = self.get_transition_vector(
				self.generate_state_t(), self.generate_action_t(),
				self.generate_reward_t(), self.generate_state_tp1()
			)
			self.add_transition_vector_to_buffer(transition_vector)

	def test_buffer_batch_getting(self):
		for _ in range(3):
			batch = self.get_batch()
			print("Current batch:")
			print(batch)

	def add_transition_vector_to_buffer(self, transition_vector: pd.DataFrame):
		self.buffer = pd.concat([self.buffer, transition_vector], ignore_index = True)

	def get_transition_vector(self, state_t: dict, action_t: float, reward_t: float, state_tp1: dict):
		transition_vector = {"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}
		transition_vector["state_t"].append(self.get_converted_env_state_vector(state_t))
		transition_vector["action_t"].append(action_t)
		transition_vector["reward_t"].append(reward_t)
		transition_vector["state_tp1"].append(self.get_converted_env_state_vector(state_tp1))

		return pd.DataFrame(transition_vector)

	def get_converted_env_state_vector(self, env_state_vector: dict):
		converted_vector = []
		for vector_component_values in env_state_vector.values():
			converted_vector += vector_component_values

		return converted_vector

	def get_batch(self) -> pd.DataFrame:
		return self.buffer.sample(self.batch_size, axis = 0)

	def generate_state_t(self):
		state_t = {
			"p": [randint(-10, 10), randint(-10, 10), randint(0, 10)],
			"v": [randint(0, 20), randint(0, 20), randint(0, 20)],
			"q": [randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)],
			"omega": [randint(0, 5), randint(0, 5), randint(0, 5)]
		}
		return state_t

	def generate_action_t(self):
		return randint(-10, 10) / 10

	def generate_reward_t(self):
		return randint(-80, 80)

	def generate_state_tp1(self):
		state_tp1 = {
			"p": [randint(-10, 10), randint(-10, 10), randint(0, 10)],
			"v": [randint(0, 20), randint(0, 20), randint(0, 20)],
			"q": [randint(-5, 5), randint(-5, 5), randint(-5, 5), randint(-5, 5)],
			"omega": [randint(0, 5), randint(0, 5), randint(0, 5)]
		}
		return state_tp1

if __name__ == "__main__":
	buffer = ReplayBuffer(
		pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}),
		25, 3
	)
	buffer.test_buffer_on_random_vectors()
	buffer.test_buffer_batch_getting()