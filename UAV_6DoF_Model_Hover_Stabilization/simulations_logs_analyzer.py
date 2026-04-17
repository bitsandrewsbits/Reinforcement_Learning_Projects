# Simulations logs analyzer: 
#	max-z(height in m) per episode,
#	mean reward per episode,
#	steps amount per episode
import os
import pandas as pd
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

import global_vars as gv

class Simulations_Logs_Analyzer:
	def __init__(self, replay_buffers_dir_path: str, p_vectors_dir_path: str):
		self.replay_buffers_dir_path = replay_buffers_dir_path
		self.p_vectors_files_dir_path = p_vectors_dir_path
		self.p_vectors_files_amount = len(os.listdir(self.p_vectors_files_dir_path))
		self.episodes_numbers = [i for i in range(1, self.p_vectors_files_amount + 1)]
		self.mean_z_values_per_episodes = []
		self.mean_reward_values_per_episodes = []
		self.steps_per_episodes = []
		self.Tsum_per_episodes = []

	def main(self):
		self.define_mean_z_values_per_episodes()
		self.define_mean_reward_steps_Tsum_per_episodes()
		
		fig, ax = plt.subplots(4, 1)
		self.plot_mean_z_per_episodes()
		self.plot_mean_reward_per_episodes()
		self.plot_steps_amount_per_episodes()
		self.plot_Tsum_per_episodes()
		plt.show()

	def plot_mean_z_per_episodes(self):
		plt.subplot(4, 1, 1)
		plt.plot(self.episodes_numbers, self.mean_z_values_per_episodes, marker = ".", label = "mean z(height)")
		plt.xlabel("episode")
		plt.ylabel("mean z(height in m)")
		plt.legend()
		plt.grid()

	def plot_Tsum_per_episodes(self):
		plt.subplot(4, 1, 2)
		plt.plot(self.episodes_numbers, self.Tsum_per_episodes, marker = ".", label = "mean Tsum")
		plt.xlabel("episode")
		plt.ylabel("mean Tsum")
		plt.legend()
		plt.grid()

	def plot_mean_reward_per_episodes(self):
		plt.subplot(4, 1, 3)
		plt.plot(self.episodes_numbers, self.mean_reward_values_per_episodes, marker = ".", label = "reward")
		plt.xlabel("episode")
		plt.ylabel("reward")
		plt.legend()
		plt.grid()

	def plot_steps_amount_per_episodes(self):
		plt.subplot(4, 1, 4)
		plt.plot(self.episodes_numbers, self.steps_per_episodes, marker = ".", label = "steps amount")
		plt.xlabel("episode")
		plt.ylabel("steps amount")
		plt.legend()
		plt.grid()

	def define_mean_z_values_per_episodes(self):
		for i in self.episodes_numbers:
			p_vector_file_path = f"{gv.P_VECTORS_IN_TIME_DIR_PATH}/{gv.P_VECTORS_FILENAME_PATTERN}{i}.jsonl"
			episode_dataset = self.get_dataset_from_JSONL(p_vector_file_path)
			episode_dataset_mean_z_value = self.get_mean_z_value_from_dataset(episode_dataset)
			self.mean_z_values_per_episodes.append(
				episode_dataset_mean_z_value
			)

	def define_mean_reward_steps_Tsum_per_episodes(self):
		for i in self.episodes_numbers:
			buffer_file_path = f"{gv.REPLAY_BUFFERS_DIR_PATH}/{gv.REPLAY_BUFFERS_FILENAME_PATTERN}{i}.jsonl"
			episode_dataset = self.get_dataset_from_JSONL(buffer_file_path)
			episode_dataset_mean_reward = self.get_mean_reward_value_from_dataset(episode_dataset)
			self.mean_reward_values_per_episodes.append(
				episode_dataset_mean_reward
			)
			episode_steps = self.get_steps_amount_from_dataset(episode_dataset)
			self.steps_per_episodes.append(episode_steps)
			self.Tsum_per_episodes.append(
				self.get_mean_Tsum_from_dataset(episode_dataset)
			)

	def get_mean_z_value_from_dataset(self, p_vectors_dataset: pd.DataFrame):
		return p_vectors_dataset["z"].mean()

	def get_mean_Tsum_from_dataset(self, replay_buffer_dataset: pd.DataFrame):
		action_t = replay_buffer_dataset['action_t'].values
		Tsums = [action[0] for action in action_t]
		return sum(Tsums) / len(Tsums)

	def get_mean_reward_value_from_dataset(self, replay_buffer_dataset: pd.DataFrame):
		return replay_buffer_dataset["reward_t"].mean()

	def get_steps_amount_from_dataset(self, replay_buffer_dataset: pd.DataFrame):
		return replay_buffer_dataset.shape[0]

	def get_dataset_from_JSONL(self, file_path: str):
		return pd.read_json(
			file_path, orient = 'records', lines = True
		)

if __name__ == "__main__":
	p_vectors_analyzer = Simulations_Logs_Analyzer(
		gv.REPLAY_BUFFERS_DIR_PATH,
		gv.P_VECTORS_IN_TIME_DIR_PATH
	)
	p_vectors_analyzer.main()