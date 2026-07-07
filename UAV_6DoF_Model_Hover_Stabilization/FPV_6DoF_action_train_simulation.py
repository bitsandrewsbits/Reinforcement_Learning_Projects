# main simulation class with SAC algorithm

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import time

import UAV_6DoF_model_environment as uavme
import UAV_1D_model_environment as u1dme
import replay_buffer as rb
import actor_NN as act
import critic_NNs as crit
import additional_functions as adf
import global_vars as gv

class UAV_6DoF_Action_Simulation:
	def __init__(self, uav_action_name: str, buffer_batch_size: int, steps_in_episode: int, 
episodes: int, test_env = "6DoF_model"):
		self.test_env = test_env
		self.uav_action_name = uav_action_name
		self.create_dirs_for_simulations(self.uav_action_name)
		self.steps_in_episode = steps_in_episode
		self.episodes = episodes
		self.buffer_batch_size = buffer_batch_size
		self.success_transitions_batch_size_percents = 0.5
		self.success_transitions_batch_size = int(
			self.success_transitions_batch_size_percents * self.buffer_batch_size
		)
		self.train_batch_size = self.buffer_batch_size - self.success_transitions_batch_size
		self.passed_steps_amount_before_learning = gv.PASSED_STEPS_BEFORE_LEARNING
		self.train_buffer_size = 10 ** 5
		self.success_transitions_buffer_size = 10 ** 4
		self.train_replay_buffer = rb.ReplayBuffer(
			self.train_buffer_size, self.buffer_batch_size
		)
		self.success_transitions_buffer = rb.ReplayBuffer(
			self.success_transitions_buffer_size, self.success_transitions_batch_size
		)
		self.current_episode_buffer = object
		self.evaluation_episode_buffer = object
		self.total_passed_steps = 0
		self.actor_training_started = False
		self.uav_model_env = self.get_uav_environment()
		self.uav_model_evaluation_env = self.get_uav_environment()
		self.uav_model_evaluation_env.evaluation_simulation = True
		self.actor = act.Actor(self.test_env)
		self.critic = crit.Critic()
		self.target_critic = crit.Critic()
		self.initialize_actor_NN_weights()
		self.initialize_critic_NNs_target_critic_NNs_weights()
		self.tau = 0.004
		self.update_critic_NNs_period_in_steps = 5
		self.update_actor_NN_period_in_steps = 10
		self.total_NNs_updates_amount = 0
		self.gamma = self.uav_model_env.discount_gamma
		self.alpha = 0.2
		self.actor_learning_rate = 1 * 10 ** (-4)
		self.critic_learning_rate = 1 * 10 ** (-4)
		self.critic_optimizer = Adam(self.critic_learning_rate)
		self.actor_optimizer = Adam(self.actor_learning_rate)
		self.action_values = self.uav_model_env.action_space_values
		self.action_evaluation_values = self.uav_model_evaluation_env.action_space_values

		self.total_simulations_time = 0
		self.replay_buffer_filename_pattern = gv.REPLAY_BUFFERS_FILENAME_PATTERN

		self.success_hover_heights_per_episode = {"success_hover_heights_amount": []}
		self.uav_action_success_trajectories_sequence_length = 0
		self.target_success_trajectories_sequence_length = 10
		self.agent_learned_uav_action = False
		self.success_trained_actor_NN_weights_path = (
			f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][3]}/{gv.SUCCESS_TRAINED_ACTOR_NN_WEIGHTS}"
		)
		self.success_trained_critic_NNs_weights_path = (
			f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][4]}/{gv.SUCCESS_TRAINED_CRITIC_NNS_WEIGHTS}"
		)
		self.success_target_critic_NNs_weights_path = (
			f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][4]}/{gv.SUCCESS_TARGET_CRITIC_NNS_WEIGHTS}"
		)

	def get_uav_environment(self):
		if self.test_env == "6DoF_model":
			return uavme.UAV_6DoF_Model_Environment(
				test_uav_action, 0.99,
				{"alpha_xy": 0.5, "alpha_z": 15, "beta": 10, "gamma": 0.05, "delta": 10 ** (-5)},
				{
					"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
					"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
				},
				{
					"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]},
					"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
				},
				0.5, 0.1, 0.05, episode_steps
			)
		elif self.test_env == "1D_model":
			return u1dme.UAV_1D_Model_Environment(
				test_uav_action,
				0.99, {"alpha_z": 5, "beta": 0.5, "delta": 2},
				{
					"z": np.array([0.481]), "vz": np.array([0])
				},
				0.5, 8, 0.02, episode_steps
			)
		else:
			raise Exception(f"[WARN]Error! You test_env={test_env} doesn't exist in simulator!")

	def main(self):
		self.load_successfully_trained_actor_NN_weights_if_exist()
		self.load_successfully_trained_critic_NNs_weights_if_exist()
		self.load_success_target_critic_NNs_weights_if_exist()
		start_time = time.time()
		for i in range(1, self.episodes + 1):
			if self.agent_learned_uav_action:
				print(f"[SUCCESS] Agent successfully learned action - {self.uav_action_name} !")
				break
			self.current_episode_buffer = rb.ReplayBuffer(self.steps_in_episode, 32)
			self.start_simulation(i)
			self.success_hover_heights_per_episode["success_hover_heights_amount"].append(
				self.uav_model_env.hover_steps_amount
			)
			if self.uav_model_env.success_stage_trajectory and self.actor_training_started:
				self.uav_action_success_trajectories_sequence_length += 1
				self.save_successfully_trained_actor_NN_weights()
				self.save_successfully_trained_critics_NNs_weights()
				self.alpha = self.alpha * 0.95
				self.run_evaluation_simulations(actor_success_trajectory = True)
			else:
				self.uav_action_success_trajectories_sequence_length = 0

			self.uav_model_env.reset_environment()
			if self.uav_action_success_trajectories_sequence_length > self.target_success_trajectories_sequence_length:
				self.agent_learned_uav_action = True
			else:
				self.agent_learned_uav_action = False

		if self.uav_action_name == "hover_stabilization":
			adf.save_dataset_to_JSONL(
				pd.DataFrame(self.success_hover_heights_per_episode),
				gv.SUCCESS_HOVER_HEIGHTS_DATASET_PATH
			)
		adf.save_dataset_to_JSONL(
			self.train_replay_buffer.buffer, f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][2]}/Train_Buffer.jsonl"
		)
		adf.save_dataset_to_JSONL(
			self.success_transitions_buffer.buffer, f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][2]}/Success_Buffer.jsonl"
		)
		end_time = time.time()
		self.total_simulations_time = end_time - start_time
		print(f"[INFO] Total simulations time in sec = {self.total_simulations_time}")
		print(f"[INFO] Total passed steps = {self.total_passed_steps}")

	def start_simulation(self, episode_number: int):
		print(f"Running Episode #{episode_number}...")
		for step_i in range(self.steps_in_episode):
			if self.uav_model_env.episode_termination:
				break
			if self.total_passed_steps > self.passed_steps_amount_before_learning:
				self.actor_training_started = True
				buffer_batch = self.train_replay_buffer.get_batch(self.success_transitions_buffer)
				states, actions, rewards, dones, next_states = self.train_replay_buffer.get_states_actions_rewards_dones_next_states(buffer_batch)
				
				if step_i % self.update_critic_NNs_period_in_steps == 0:
					next_states = tf.cast(next_states, tf.float32)
					next_actions, log_actions_batch_probabilities = self.actor(next_states)
					states_actions_batch = self.critic.get_states_actions_batch(states, actions)
					next_states_actions_batch = self.critic.get_states_actions_batch(next_states, next_actions)
					self.update_critic_Q1_Q2_trainable_weights(
						states_actions_batch, next_states_actions_batch, rewards,
						dones, log_actions_batch_probabilities
					)
					self.update_target_critic_NNs_weights()
				if step_i % self.update_actor_NN_period_in_steps == 0:
					self.total_NNs_updates_amount += 1
					self.update_actor_trainable_weights(states)

			current_state = self.uav_model_env.current_state_vector
			current_state = tf.expand_dims(adf.get_values_as_array(current_state), axis = 0)
			action_t, _ = self.actor(current_state)
			self.action_values["T_sum"] = self.get_action_Tsum_in_hover_representation(action_t)
			if self.test_env == "6DoF_model":
				self.action_values["tau_x"], self.action_values["tau_y"], self.action_values["tau_z"] = self.get_scaled_moments_tau(action_t)
			next_state, reward_t, done = self.uav_model_env.step(self.action_values)
			transition_vector = self.train_replay_buffer.get_transition_vector(
				current_state, self.action_values, reward_t, done, next_state
			)
			
			self.total_passed_steps += 1
			self.train_replay_buffer.update_buffer_with_new_transition_vector(transition_vector, self.total_passed_steps)
			self.current_episode_buffer.add_transition_vector_to_buffer(transition_vector)

			if self.uav_model_env.transition_vector_is_success(transition_vector):
				self.success_transitions_buffer.update_success_buffer_with_new_transition(transition_vector)

			if self.total_NNs_updates_amount % gv.RUN_EVAL_SIMULATIONS_EVERY_ACTOR_UPDATES == 0 and self.total_NNs_updates_amount > 0:
				self.run_evaluation_simulations()

		current_episode_buffer_path = f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][2]}/{self.replay_buffer_filename_pattern}{episode_number}.jsonl"
		adf.save_dataset_to_JSONL(self.current_episode_buffer.buffer, current_episode_buffer_path)
		self.uav_model_env.save_episode_p_vectors_to_JSONL(episode_number)

	def run_evaluation_simulations(self, actor_success_trajectory = False):
		if actor_success_trajectory:
			evaluation_simulations_buffers_path = (
				f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][6]}/"
				f"{gv.EVAL_SIMULATIONS_AFTER_ACTOR_SUCCESS_TRAJECTORIES_DIR_PATTERN}"
				f"{self.uav_action_success_trajectories_sequence_length}_success_trajectories"
			)
		else:
			evaluation_simulations_buffers_path = (
				f"{gv.UAV_ACTIONS_SIMULATIONS_DIRS[self.uav_action_name][5]}/"
				f"{gv.EVAL_SIMULATIONS_N_ACTOR_UPDATES_EPISODE_DIR_PATTERN}"
				f"{self.total_NNs_updates_amount}_actor_updates"
			)
		if not os.path.exists(evaluation_simulations_buffers_path):
			os.mkdir(evaluation_simulations_buffers_path)
		else:
			return -1
		for eval_simulation_number in range(1, 10 + 1):
			self.start_evaluation_episode(evaluation_simulations_buffers_path, eval_simulation_number)
			self.uav_model_evaluation_env.reset_environment()

	def start_evaluation_episode(self, evaluation_simulations_buffers_path, eval_simulation_number: int):
		self.evaluation_episode_buffer = rb.ReplayBuffer(self.steps_in_episode, 32)
		for step_i in range(self.steps_in_episode):
			if self.uav_model_evaluation_env.episode_termination:
				break
			current_state = self.uav_model_evaluation_env.current_state_vector
			current_state = tf.expand_dims(adf.get_values_as_array(current_state), axis = 0)
			action_t = self.actor(current_state, evaluation_mode = True)
			self.action_evaluation_values["T_sum"] = self.get_action_Tsum_in_hover_representation(action_t)
			if self.test_env == "6DoF_model":
				self.action_evaluation_values["tau_x"], self.action_evaluation_values["tau_y"], self.action_evaluation_values["tau_z"] = self.get_scaled_moments_tau(action_t)
			next_state, reward_t, done = self.uav_model_evaluation_env.step(self.action_evaluation_values)
			transition_vector = self.evaluation_episode_buffer.get_transition_vector(
				current_state, self.action_evaluation_values, reward_t, done, next_state
			)
			self.evaluation_episode_buffer.add_transition_vector_to_buffer(transition_vector)
		evaluation_simulation_buffer_path = f"{evaluation_simulations_buffers_path}/simulation-{eval_simulation_number}.jsonl"
		adf.save_dataset_to_JSONL(self.evaluation_episode_buffer.buffer, evaluation_simulation_buffer_path)

	def get_action_Tsum_in_hover_representation(self, action_Tsum: tf.Tensor):
		action_Tsum_float = tf.keras.ops.convert_to_numpy(action_Tsum)[0][0]
		updated_Tsum = self.uav_model_env.hover_action_components["T_sum"] * (1 + 0.02 * action_Tsum_float)

		return updated_Tsum

	def get_scaled_moments_tau(self, action: tf.Tensor):
		scaled_tau_x = self.uav_model_env.max_tau_x * action[1]
		scaled_tau_y = self.uav_model_env.max_tau_y * action[2]
		scaled_tau_z = self.uav_model_env.max_tau_z * action[3]
		return scaled_tau_x, scaled_tau_y, scaled_tau_z

	def update_critic_Q1_Q2_trainable_weights(self, states_actions_batch, next_states_actions_batch,
	rewards_batch, dones_batch, log_actions_batch_probabilities):
		with tf.GradientTape() as grad:
			Q1_batch, Q2_batch = self.critic(states_actions_batch)
			target_Q1_batch, target_Q2_batch = self.target_critic(next_states_actions_batch)
			min_target_Q1_Q2 = tf.math.minimum(target_Q1_batch, target_Q2_batch)
			rewards = rewards_batch.reshape(len(rewards_batch), 1)
			target_Q = tf.stop_gradient(
				rewards + (1 - dones_batch) * self.gamma * (min_target_Q1_Q2 - self.alpha * log_actions_batch_probabilities)
			)
			critic_loss = tf.math.reduce_mean(
				(Q1_batch - target_Q) ** 2 + (Q2_batch - target_Q) ** 2
			)
		loss_grad = grad.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(loss_grad, self.critic.trainable_variables))

	def update_actor_trainable_weights(self, states_n_batch):
		with tf.GradientTape() as grad:
			actions_n_batch, log_actions_batch_probabilities = self.actor(states_n_batch)
			actions_n_batch = tf.cast(actions_n_batch, tf.float32)
			states_n_batch = tf.cast(states_n_batch, tf.float32)
			states_actions_batch = self.critic.get_states_actions_batch(states_n_batch, actions_n_batch)
			Q1, Q2 = self.critic(states_actions_batch)
			min_Q1_Q2_batch = tf.math.minimum(Q1, Q2)
			actor_loss = tf.math.reduce_mean(
				self.alpha * log_actions_batch_probabilities - min_Q1_Q2_batch
			)
		loss_grad = grad.gradient(actor_loss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(loss_grad, self.actor.trainable_variables))

	def initialize_actor_NN_weights(self):
		states_actions_batch_shape = (1, self.uav_model_env.current_state_vector_size)
		zeros_states_actions = np.zeros(states_actions_batch_shape)
		self.actor(zeros_states_actions)

	def initialize_critic_NNs_target_critic_NNs_weights(self):
		action_space_size = len(self.uav_model_env.action_space_values.values())
		states_actions_batch_shape = (1, self.uav_model_env.current_state_vector_size + action_space_size)
		zeros_states_actions = np.zeros(states_actions_batch_shape)
		self.critic(zeros_states_actions)
		self.target_critic(zeros_states_actions)
		for (target_critic_var, critic_var) in zip(self.target_critic.variables, self.critic.variables):
			target_critic_var.assign(critic_var)

	def update_target_critic_NNs_weights(self):
		for (target_critic_var, critic_var) in zip(self.target_critic.variables, self.critic.variables):
			target_critic_var.assign(
				self.tau * critic_var + (1 - self.tau) * target_critic_var
			)

	def create_dirs_for_simulations(self, action_name: str):
		simulations_data_dirs = gv.UAV_ACTIONS_SIMULATIONS_DIRS[action_name]
		for sim_dir in simulations_data_dirs:
			adf.create_dir_if_need(sim_dir)

	def save_successfully_trained_actor_NN_weights(self):
		self.actor.save_weights(
			self.success_trained_actor_NN_weights_path
		)

	def save_successfully_trained_critics_NNs_weights(self):
		self.critic.save_weights(
			self.success_trained_critic_NNs_weights_path
		)
		self.target_critic.save_weights(
			self.success_target_critic_NNs_weights_path
		)

	def load_successfully_trained_actor_NN_weights_if_exist(self):
		if os.path.exists(self.success_trained_actor_NN_weights_path):
			print("[INFO] Loading last successfully trained actor NN weights...")
			self.actor.load_weights(self.success_trained_actor_NN_weights_path)

	def load_successfully_trained_critic_NNs_weights_if_exist(self):
		if os.path.exists(self.success_trained_critic_NNs_weights_path):
			print("[INFO] Loading last successfully trained critic NNs weights...")
			self.critic.load_weights(self.success_trained_critic_NNs_weights_path)

	def load_success_target_critic_NNs_weights_if_exist(self):
		if os.path.exists(self.success_target_critic_NNs_weights_path):
			print("[INFO] Loading last success target critic NNs weights...")
			self.target_critic.load_weights(self.success_target_critic_NNs_weights_path)

if __name__ == "__main__":
	episode_steps = 2000
	buffer_batch_size = 256
	test_env = "1D_model"
	test_uav_action = "hover_stabilization"
	fpv_hover_simulation = UAV_6DoF_Action_Simulation(
		test_uav_action, buffer_batch_size,
		episode_steps, 500, test_env
	)
	fpv_hover_simulation.main()