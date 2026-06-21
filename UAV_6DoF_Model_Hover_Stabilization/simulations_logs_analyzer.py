# Simulations logs analyzer: 
import os
import pandas as pd
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt

import global_vars as gv
import additional_functions as adf
import argparse

class Simulations_Logs_Analyzer:
	def __init__(self):
		self.selected_uav_action_name = ""
		self.possible_uav_actions = ["ascent", "hover_stabilization"]
		self.possible_eval_simul_every_actor_updates_nums = []
		self.replay_buffers_dir_path = "."
		self.p_vectors_files_dir_path = "."
		self.p_vectors_files_amount = 0
		self.evaluation_episodes_amount = len(os.listdir(gv.UAV_ACTIONS_SIMULATIONS_DIRS["hover_stabilization"][5]))
		self.uav_action_eval_simulations_amount = [
			i * gv.RUN_EVALUATION_SIMULATIONS_EVERY_ACTOR_UPDATES for i in range(1, self.evaluation_episodes_amount + 1)
		]
		self.episodes_numbers = []
		self.success_hover_heights_amount_per_episodes = []
		self.episodes_rewards = []
		self.max_height_z_per_episodes = []
		self.steps_per_episodes = []
		self.Tsum_per_episodes = []
		self.check_episode_period = 10
		self.battery_mass_in_kg = 0.9
		self.UAV_motor_mass_in_kg = 0.06
		self.UAV_mass_in_kg = self.battery_mass_in_kg + 4 * self.UAV_motor_mass_in_kg
		self.T_hover = self.UAV_mass_in_kg * 9.8

	def main(self):
		# TODO: think, how to run this script with arguments for diff show-operations.
		# Main arguments: 
		# 1) show all uav actions logs stats
		# 2) show uav action logs stats changing per episodes
		# 3) show uav one evaluation simulations in one plot for comparison.
		plots_modes_parser = argparse.ArgumentParser()
		plots_modes_parser.add_argument(
			"-act", "--uav_action",
			choices = ["ascent", "hover_stabilization"],
			help = "UAV action for simulation."
		)
		plots_modes_parser.add_argument(
			"-auass", "--all_uav_actions_statistics",
        	help = "Show all uav actions main plots sequentialy."
		)
		plots_modes_parser.add_argument(
			"-oalpe", "--one_action_logs_per_episodes",
			help = "Show one uav action logs plots per some episodes"
		)
		plots_modes_parser.add_argument(
			"-oaeeau", "--one_action_eval_episode_after_N_actor_updates",
			choices = self.uav_action_eval_simulations_amount,
			help = f"Show one action plot with evaluation trajectories after N actor updates."
		)
		received_args = plots_modes_parser.parse_args()
		if received_args.uav_action:
			self.selected_uav_action_name = received_args.uav_action
			self.update_all_variables_for_uav_action(self.selected_uav_action_name)
		elif received_args.all_uav_actions_statistics:
			fig, ax = plt.subplots(4, 1)
			for uav_action in self.possible_uav_actions:
				self.update_all_variables_for_uav_action(uav_action)
				self.define_episodes_rewards_Tsum_per_episodes()
				if uav_action == "ascent":
					self.show_plots_for_ascent_action()
				elif uav_action == "hover_stabilization":
					self.show_plots_for_hover_stabilization()
				self.plot_episode_reward_per_episodes()
				self.plot_steps_amount_per_episodes()
				self.plot_Tsum_per_episodes()
				plt.show()
		if received_args.one_action_logs_per_episodes:
			for check_episode_number in range(1, len(self.episodes_numbers) + 1):
				if check_episode_number % self.check_episode_period == 0:
					self.plot_dependencies_with_t(
						[
							("state_t", 0, "z"), ("state_t", 1, "vz"),
							("action_t", 0, "T_sum"), ("reward_t", -1, "reward_t")
						],
						check_episode_number
					)
		if received_args.one_action_evaluation_simulations:
			pass
			# TODO: finish uav-action evaluation simulations plot via command

	def update_all_variables_for_uav_action(self, uav_action: str):
		self.replay_buffers_dir_path = gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action][2]
		self.p_vectors_files_dir_path = gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action][1]
		self.p_vectors_files_amount = len(os.listdir(self.p_vectors_files_dir_path)) - 1
		self.episodes_numbers = [i for i in range(1, self.p_vectors_files_amount + 1)]
		self.episodes_rewards = []
		self.steps_per_episodes = []
		self.Tsum_per_episodes = []

	def show_plots_for_ascent_action(self):
		self.define_max_height_z_per_episodes()
		self.plot_max_height_z_per_episodes()

	def show_plots_for_hover_stabilization(self):
		self.define_success_hover_heights_per_episodes()
		self.plot_success_hover_heights_z_per_episodes()

	def plot_max_height_z_per_episodes(self):
		plt.subplot(4, 1, 1)
		plt.scatter(
			self.episodes_numbers, self.max_height_z_per_episodes,
			marker = '.', label = "max-height-z"
		)
		plt.xlabel("episode")
		plt.ylabel("max_height_z")
		plt.legend()
		plt.grid()

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

	def plot_episode_reward_per_episodes(self):
		print(len(self.episodes_numbers))
		print(len(self.episodes_rewards))
		plt.subplot(4, 1, 3)
		plt.scatter(
			self.episodes_numbers, self.episodes_rewards,
			marker = '.', label = "reward"
		)
		plt.xlabel("episode")
		plt.ylabel("episode reward")
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

	def define_episodes_rewards_Tsum_per_episodes(self):
		for i in self.episodes_numbers:
			buffer_file_path = f"{self.replay_buffers_dir_path}/{gv.REPLAY_BUFFERS_FILENAME_PATTERN}{i}.jsonl"
			episode_dataset = adf.get_dataset_from_JSONL(buffer_file_path)
			episode_dataset_episode_reward = self.get_episode_reward_value_from_dataset(episode_dataset)
			self.episodes_rewards.append(
				episode_dataset_episode_reward
			)
			episode_steps = self.get_steps_amount_from_dataset(episode_dataset)
			self.steps_per_episodes.append(episode_steps)
			self.Tsum_per_episodes.append(
				self.get_mean_Tsum_from_dataset(episode_dataset)
			)

	def define_max_height_z_per_episodes(self):
		for i in self.episodes_numbers:
			buffer_file_path = f"{self.p_vectors_files_dir_path}/{gv.P_VECTORS_FILENAME_PATTERN}{i}.jsonl"
			episode_dataset = adf.get_dataset_from_JSONL(buffer_file_path)
			episode_dataset_max_z = self.get_max_z_value_from_dataset(episode_dataset)
			self.max_height_z_per_episodes.append(
				episode_dataset_max_z
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

	def get_episode_reward_value_from_dataset(self, replay_buffer_dataset: pd.DataFrame):
		return replay_buffer_dataset["reward_t"].sum()

	def get_steps_amount_from_dataset(self, replay_buffer_dataset: pd.DataFrame):
		return replay_buffer_dataset.shape[0]

	def plot_dependencies_with_t(self, vector_names_indexes_component: list[tuple], episode_number: int):
		plot_amount = len(vector_names_indexes_component)
		fig, ax = plt.subplots(plot_amount, 1)
		plot_index = 1
		for vector_name_index_component in vector_names_indexes_component:
			plt.subplot(plot_amount, 1, plot_index)
			self.plot_episode_vector_component_values_with_t(vector_name_index_component, episode_number)
			plot_index += 1
		plt.show()

	def plot_episode_vector_component_values_with_t(self, vector_name_index_component: tuple, episode_number: int):
		vector_name = vector_name_index_component[0]
		vector_component_index = vector_name_index_component[1]
		state_vector_component_name = vector_name_index_component[2]
		vector_component_values = self.get_vector_component_from_buffer(vector_name, vector_component_index, episode_number)
		time_steps = [round(t / 100, 2) for t in range(len(vector_component_values))]
		plt.title(f"Episode-{episode_number}, {state_vector_component_name}(t)")
		plt.scatter(
			time_steps, vector_component_values,
			marker = '.', label = f"{state_vector_component_name}(t)"
		)
		plt.xlabel("time, sec")
		plt.ylabel(f"{state_vector_component_name}, m")
		plt.legend()
		plt.grid()

	def get_vector_component_from_buffer(self, vector_name: str, vector_component_index: int, episode_number: int):
		buffer_file_path = f"{self.replay_buffers_dir_path}/{gv.REPLAY_BUFFERS_FILENAME_PATTERN}{episode_number}.jsonl"
		episode_dataset = adf.get_dataset_from_JSONL(buffer_file_path)
		vector_component_values = []
		for state_vector in episode_dataset[vector_name]:
			if type(state_vector) == list:
				vector_component_values.append(state_vector[vector_component_index])
			else:
				transition_vector_value = state_vector
				vector_component_values.append(transition_vector_value)

		return vector_component_values

	def get_vector_component_name(self, vector_name: str, vector_component_index: int):
		if vector_name == "state_t":
			return self.index_to_state_vector_component[vector_component_index]
		elif vector_name == "action_t":
			return self.index_to_action_t_component[vector_component_index]
		else:
			return "reward_t"

if __name__ == "__main__":
	# 1)
	# for uav_action in uav_actions:
	# 	p_vectors_analyzer_uav_action = Simulations_Logs_Analyzer(
	# 		uav_action,
	# 		gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action][2],
	# 		gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action][1]
	# 	)
	# 	p_vectors_analyzer_uav_action.main()
	
	# # 2)
	# p_vectors_analyzer_uav_action = Simulations_Logs_Analyzer(
	# 	"hover_stabilization",
	# 	gv.UAV_ACTIONS_SIMULATIONS_DIRS["hover_stabilization"][2],
	# 	gv.UAV_ACTIONS_SIMULATIONS_DIRS["hover_stabilization"][1]
	# )
	# check_episodes_amount = p_vectors_analyzer_uav_action.p_vectors_files_amount
	# for check_episode_number in range(1, check_episodes_amount + 1):
	# 	if check_episode_number % 10 == 0:
	# 		p_vectors_analyzer_uav_action.plot_dependencies_with_t(
	# 			[
	# 				("state_t", 0, "z"), ("state_t", 1, "vz"),
	# 				("action_t", 0, "T_sum"), ("reward_t", -1, "reward_t")
	# 			],
	# 			check_episode_number
	# 		)
	simulations_logs_analyzer = Simulations_Logs_Analyzer()
	simulations_logs_analyzer.main()
	# 3)