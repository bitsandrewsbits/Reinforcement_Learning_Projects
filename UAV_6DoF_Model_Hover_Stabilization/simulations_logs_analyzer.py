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
import additional_functions as adf

class Simulations_Logs_Analyzer:
	def __init__(self, replay_buffers_dir_path: str, p_vectors_dir_path: str):
		self.replay_buffers_dir_path = replay_buffers_dir_path
		self.p_vectors_files_dir_path = p_vectors_dir_path
		self.p_vectors_files_amount = len(os.listdir(self.p_vectors_files_dir_path)) - 1
		self.episodes_numbers = [i for i in range(1, self.p_vectors_files_amount + 1)]
		self.success_hover_heights_amount_per_episodes = []
		self.mean_reward_values_per_episodes = []
		self.steps_per_episodes = []
		self.Tsum_per_episodes = []
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.T_hover = 1.28 * self.UAV_mass_in_kg * 9.8

	def main(self):
		self.define_success_hover_heights_per_episodes()
		self.define_mean_reward_steps_Tsum_per_episodes()
		
		fig, ax = plt.subplots(4, 1)
		self.plot_success_hover_heights_z_per_episodes()
		self.plot_mean_reward_per_episodes()
		self.plot_steps_amount_per_episodes()
		self.plot_Tsum_per_episodes()
		plt.show()

	def plot_success_hover_heights_z_per_episodes(self):
		plt.subplot(4, 1, 1)
		plt.scatter(
			self.episodes_numbers, self.success_hover_heights_amount_per_episodes,
			marker = '.', label = "hover-z-amount"
		)
		plt.xlabel("episode")
		plt.ylabel("success_hover_z_amount")
		plt.legend()
		plt.grid()

	def plot_Tsum_per_episodes(self):
		plt.subplot(4, 1, 2)
		plt.scatter(
			self.episodes_numbers, self.Tsum_per_episodes,
			marker = '.', label = "mean Tsum"
		)
		plt.plot(
			self.episodes_numbers, [self.T_hover for _ in self.Tsum_per_episodes],
			color = 'red', label = "T-hover"
		)
		plt.xlabel("episode")
		plt.ylabel("mean Tsum")
		plt.legend()
		plt.grid()

	def plot_mean_reward_per_episodes(self):
		plt.subplot(4, 1, 3)
		plt.scatter(
			self.episodes_numbers, self.mean_reward_values_per_episodes,
			marker = '.', label = "reward"
		)
		plt.xlabel("episode")
		plt.ylabel("reward")
		plt.legend()
		plt.grid()

	def plot_steps_amount_per_episodes(self):
		plt.subplot(4, 1, 4)
		plt.scatter(
			self.episodes_numbers, self.steps_per_episodes,
			marker = '.', label = "steps amount"
		)
		plt.xlabel("episode")
		plt.ylabel("steps amount")
		plt.legend()
		plt.grid()

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

	def define_success_hover_heights_per_episodes(self):
		dataset_success_hover_z = adf.get_dataset_from_JSONL(gv.SUCCESS_HOVER_HEIGHTS_DATASET_PATH)
		self.success_hover_heights_amount_per_episodes = dataset_success_hover_z["success_hover_heights_amount"].values

	def get_mean_z_value_from_dataset(self, p_vectors_dataset: pd.DataFrame):
		return p_vectors_dataset["z"].mean()

	def get_max_z_value_from_dataset(self, p_vectors_dataset: pd.DataFrame):
		return p_vectors_dataset["z"].max()

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