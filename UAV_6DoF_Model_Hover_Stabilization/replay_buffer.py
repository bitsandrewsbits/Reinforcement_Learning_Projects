# Replay Buffer for transition vectors saving per episode
# transition vector structure - (s_t, a_t, r_t, s_t+1)
# replay buffer structure = [(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2),..., (s_T-1, a_T-1, r_T-1, s_T)] =>
# => [[x0, y0, z0, vx0, vy0, vz0, q00, q10, q20, q30, omegax0, omegay0, omegaz0, a_0, r_0, x1, y1, z1, vx1, vy1, vz1, q01, q11, q21, q31, omegax1, omegay1, omegaz1],...]
from random import randint

class ReplayBuffer:
	def __init__(self, buffer_max_transition_vectors: int, batch_size: int):
		self.buffer_max_transition_vectors = buffer_max_transition_vectors
		self.buffer = []
		self.batch_size = batch_size
		self.batch_start_buffer_index = 0
		self.batch_end_buffer_index = self.batch_size
		self.buffer_end_by_batching = False

	def test_buffer_on_random_vectors(self):
		for _ in range(self.buffer_max_transition_vectors):
			transition_vector = self.get_transition_vector(
				self.generate_state_t(), self.generate_action_t(),
				self.generate_reward_t(), self.generate_state_tp1()
			)
			self.add_transition_vector(transition_vector)

	def test_batching(self):
		batch_number = 1
		while not self.buffer_end_by_batching:
			print({self.batch_start_buffer_index}, {self.batch_end_buffer_index})
			batch = self.get_batch()
			print(f"Batch #{batch_number}: {len(batch)}")
			batch_number += 1

	def add_transition_vector(self, transition_vector):
		self.buffer.append(transition_vector)

	def get_transition_vector(self, state_t: dict, action_t, reward_t, state_tp1: dict):
		transition_vector = []
		converted_state_t = self.get_converted_env_state_vector(state_t)
		transition_vector += converted_state_t
		transition_vector += [action_t, reward_t]
		converted_state_tp1 = self.get_converted_env_state_vector(state_tp1)
		transition_vector += converted_state_tp1

		return transition_vector

	def get_converted_env_state_vector(self, env_state_vector: dict):
		converted_vector = []
		for vector_component_values in env_state_vector.values():
			converted_vector += vector_component_values

		return converted_vector

	def get_batch(self):
		batch = self.buffer[self.batch_start_buffer_index:self.batch_end_buffer_index]
		if self.batch_end_buffer_index >= self.buffer_max_transition_vectors:
			self.buffer_end_by_batching = True
			batch = self.buffer[self.batch_start_buffer_index:]
		self.update_batch_buffer_indexes()
		return batch

	def update_batch_buffer_indexes(self):
		self.batch_start_buffer_index = self.batch_end_buffer_index
		self.batch_end_buffer_index += self.batch_size

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
	buffer = ReplayBuffer(25, 7)
	buffer.test_buffer_on_random_vectors()
	buffer.test_batching()