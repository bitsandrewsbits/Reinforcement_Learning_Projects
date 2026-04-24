# main simulation class with SAC algorithm

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time

import UAV_6DoF_model_environment as uavme
import replay_buffer as rb
import actor_NN as act
import critic_NNs as crit
import additional_functions as adf
import global_vars as gv

class FPV_Hover_Stabilization_Simulation:
	def __init__(self, uav_model_env, buffer_batch_size: int, steps_in_episode: int, 
episodes: int):
		self.create_dirs_for_simulations()
		self.steps_in_episode = steps_in_episode
		self.episodes = episodes
		self.buffer_batch_size = buffer_batch_size
		self.passed_steps_amount_before_learning = gv.PASSED_STEPS_BEFORE_LEARNING
		self.train_buffer_size = 10 ** 5
		self.train_replay_buffer = rb.ReplayBuffer(self.train_buffer_size, self.buffer_batch_size)
		self.current_episode_buffer = object
		self.total_passed_steps = 0
		self.uav_model_env = uav_model_env
		self.actor = act.Actor(self.train_replay_buffer)
		self.critic = crit.Critic()
		self.target_critic = crit.Critic()
		self.tau = 0.005
		self.update_critic_NNs_period_in_steps = 2
		self.update_actor_NN_period_in_steps = 4
		self.gamma = self.uav_model_env.discount_gamma
		self.alpha = 0.2
		self.actor_learning_rate = 1 * 10 ** (-4)
		self.critic_learning_rate = 3 * 10 ** (-4)
		self.critic_optimizer = Adam(self.critic_learning_rate)
		self.actor_optimizer = Adam(self.actor_learning_rate)
		self.init_critic_NNs_weights = True

		self.total_simulations_time = 0
		self.replay_buffer_filename_pattern = gv.REPLAY_BUFFERS_FILENAME_PATTERN

	def main(self):
		start_time = time.time()
		for i in range(1, self.episodes + 1):
			self.current_episode_buffer = rb.ReplayBuffer(self.steps_in_episode, 32)
			self.uav_model_env.reset_environment()
			self.start_simulation(i)
		adf.save_buffer_to_JSONL(self.train_replay_buffer.buffer, f"{gv.REPLAY_BUFFERS_DIR_PATH}/Train_Buffer.jsonl")
		end_time = time.time()
		self.total_simulations_time = end_time - start_time
		print(f"[INFO] Total simulations time in sec = {self.total_simulations_time}")
		print(f"[INFO] Total passed steps = {self.total_passed_steps}")

	def start_simulation(self, episode_number: int):
		for step_i in range(self.steps_in_episode):
			if self.uav_model_env.early_episode_termination:
				break
			train_buffer_size = self.train_replay_buffer.buffer.shape[0]
			if self.total_passed_steps > self.passed_steps_amount_before_learning:
				buffer_batch = self.train_replay_buffer.get_batch()
				states, actions, rewards, next_states = self.train_replay_buffer.get_states_actions_rewards_next_states(buffer_batch)
				
				if step_i % self.update_critic_NNs_period_in_steps == 0:
					next_states = tf.cast(next_states, tf.float32)
					self.actor(next_states)
					next_actions = self.actor.get_actions_n_batch()
					log_actions_batch_probabilities = self.actor.get_log_probabilities()
					states_actions_batch = self.critic.get_states_actions_batch(states, actions)
					next_states_actions_batch = self.critic.get_states_actions_batch(next_states, next_actions)
					if self.init_critic_NNs_weights:
						self.initialize_critic_NNs_target_critic_NNs_weights(states_actions_batch)
						self.init_critic_NNs_weights = False
					self.update_critic_Q1_Q2_trainable_weights(
						states_actions_batch, next_states_actions_batch, rewards, log_actions_batch_probabilities
					)
					self.update_target_critic_NNs_weights()
				if step_i % self.update_actor_NN_period_in_steps == 0:
					self.update_actor_trainable_weights(states, actions)

			current_state = self.uav_model_env.current_state_vector
			action_t = self.actor.get_action(current_state)
			# print("Action Tsum before scaling:", action_t["T_sum"])
			action_t["T_sum"] = self.get_action_Tsum_in_hover_representation(action_t["T_sum"])
			next_state, reward_t = self.uav_model_env.step(action_t)
			transition_vector = self.train_replay_buffer.get_transition_vector(
				current_state, action_t, reward_t, next_state
			)
			self.total_passed_steps += 1
			self.actor.update_total_passed_steps(self.total_passed_steps)
			self.train_replay_buffer.update_buffer_with_new_transition_vector(transition_vector, self.total_passed_steps)
			self.current_episode_buffer.add_transition_vector_to_buffer(transition_vector, step_i)
		current_episode_buffer_path = f"{gv.REPLAY_BUFFERS_DIR_PATH}/{self.replay_buffer_filename_pattern}{episode_number}.jsonl"
		adf.save_buffer_to_JSONL(self.current_episode_buffer.buffer, current_episode_buffer_path)
		self.uav_model_env.save_episode_p_vectors_to_JSONL(episode_number)

	def get_action_Tsum_in_hover_representation(self, action_Tsum):
		updated_Tsum = self.uav_model_env.hover_action_components["T_sum"] * (1 + 0.4 * action_Tsum)

		return updated_Tsum

	def update_critic_Q1_Q2_trainable_weights(self, states_actions_batch, next_states_actions_batch, rewards_batch, log_actions_batch_probabilities):
		with tf.GradientTape() as grad:
			Q1_batch, Q2_batch = self.critic(states_actions_batch)
			target_Q1_batch, target_Q2_batch = self.target_critic(next_states_actions_batch)
			min_target_Q1_Q2 = tf.math.minimum(target_Q1_batch, target_Q2_batch)
			rewards = rewards_batch.reshape(len(rewards_batch), 1)
			target_Q = tf.stop_gradient(
				rewards + self.gamma * (min_target_Q1_Q2 - self.alpha * log_actions_batch_probabilities)
			)
			critic_loss = tf.math.reduce_mean(
				(Q1_batch - target_Q) ** 2 + (Q2_batch - target_Q) ** 2
			)
			loss_grad = grad.gradient(critic_loss, self.critic.trainable_variables)
			self.critic_optimizer.apply_gradients(zip(loss_grad, self.critic.trainable_variables))

	def update_actor_trainable_weights(self, states_n_batch, actions_n_batch):
		states_actions_batch = self.critic.get_states_actions_batch(states_n_batch, actions_n_batch)
		Q1, Q2 = self.critic(states_actions_batch)
		min_Q1_Q2_batch = tf.math.minimum(Q1, Q2)
		with tf.GradientTape() as grad:
			means_batch, log_stds_batch = self.actor(states_n_batch)
			log_actions_batch_probabilities = self.actor.get_log_probabilities()
			actor_loss = tf.math.reduce_mean(
				self.alpha * log_actions_batch_probabilities - min_Q1_Q2_batch
			)
			loss_grad = grad.gradient(actor_loss, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients(zip(loss_grad, self.actor.trainable_variables))

	def initialize_critic_NNs_target_critic_NNs_weights(self, states_actions_batch):
		self.critic(states_actions_batch)
		self.target_critic(states_actions_batch)
		for (target_critic_var, critic_var) in zip(self.target_critic.variables, self.critic.variables):
			target_critic_var.assign(critic_var)

	def update_target_critic_NNs_weights(self):
		for (target_critic_var, critic_var) in zip(self.target_critic.variables, self.critic.variables):
			target_critic_var.assign(
				self.tau * critic_var + (1 - self.tau) * target_critic_var
			)

	def create_dirs_for_simulations(self):
		simulations_data_dirs = [gv.SIMULATION_DIR, gv.P_VECTORS_IN_TIME_DIR_PATH, gv.REPLAY_BUFFERS_DIR_PATH]
		for sim_dir in simulations_data_dirs:
			adf.create_dir_if_need(sim_dir)

if __name__ == "__main__":
	episode_steps = 1000
	UAV_model_environment = uavme.UAV_6DoF_Model_Environment(
		0.99, {"alpha_xy": 10 ** (-2), "alpha_z": 10, "beta": 5, "gamma": 5, "delta": 10 ** (-6)},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]},
			"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
		},
		0.5, 2, episode_steps
	)
	buffer_batch_size = 128
	fpv_hover_simulation = FPV_Hover_Stabilization_Simulation(
		UAV_model_environment, buffer_batch_size, episode_steps, 1000
	)
	fpv_hover_simulation.main()