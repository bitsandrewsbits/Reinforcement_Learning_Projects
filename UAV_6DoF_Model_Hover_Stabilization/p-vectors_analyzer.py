# p-vectors analyzer:
# 1) reading all p-vectors-in-time JSONL into dataframes
# 2) plot: X - episode number, Y - max z(height) value
import os
import pandas as pd
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

import global_vars as gv

class P_Vectors_Analyzer:
	def __init__(self, files_dir_path: str):
		self.p_vectors_files_dir_path = files_dir_path
		self.p_vectors_files = os.listdir(self.p_vectors_files_dir_path)
		self.episodes_numbers = [i for i in range(1, len(self.p_vectors_files) + 1)]
		self.max_z_values_per_episodes = []

	def main(self):
		self.define_all_max_z_values_per_episodes()
		self.plot_max_z_per_episodes()

	def plot_max_z_per_episodes(self):
		plt.plot(self.episodes_numbers, self.max_z_values_per_episodes, marker = ".", label = "max z(height)")
		plt.legend()
		plt.grid()
		plt.show()

	def define_all_max_z_values_per_episodes(self):
		for p_vector_file in self.p_vectors_files:
			current_dataset = self.get_dataset_from_JSONL(p_vector_file)
			current_dataset_max_z_value = self.get_max_z_value_from_dataset(current_dataset)
			self.max_z_values_per_episodes.append(
				current_dataset_max_z_value
			)

	def get_max_z_value_from_dataset(self, dataset: pd.DataFrame):
		return dataset["z"].max()

	def get_dataset_from_JSONL(self, file: str):
		dataset_path = f"{self.p_vectors_files_dir_path}/{file}"
		return pd.read_json(
			dataset_path, orient = 'records', lines = True
		)

if __name__ == "__main__":
	p_vectors_analyzer = P_Vectors_Analyzer(gv.P_VECTORS_IN_TIME_DIR_PATH)
	p_vectors_analyzer.main()