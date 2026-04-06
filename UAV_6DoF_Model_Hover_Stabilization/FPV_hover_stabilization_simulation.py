# main simulation class with SAC algorithm

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import UAV_6DoF_model_environment as uavme
import replay_buffer as rb
import actor_NN as act
import critic_NNs as crit
import additional_functions as adf
import global_vars as gv

class FPV_Hover_Stabilization_Simulation:
	def __init__(self, uav_model_env, buffer_batch_size: int, learning_rate, steps_in_episode: int, 
episodes: int):
		self.create_dirs_for_simulations()
		self.steps_in_episode = steps_in_episode
		self.episodes = episodes
		self.buffer_batch_size = buffer_batch_size
		self.train_replay_buffer = rb.ReplayBuffer(steps_in_episode, self.buffer_batch_size)
		self.current_episode_buffer = object

		self.uav_model_env = uav_model_env
		self.actor = act.Actor(self.train_replay_buffer)
		self.critic = crit.Critic()
		self.update_actor_critic_NNs_period_in_steps = 10
		self.gamma = self.uav_model_env.discount_gamma
		self.alpha = 0.2
		self.learning_rate = learning_rate
		self.critic_optimizer = Adam(self.learning_rate)
		self.actor_optimizer = Adam(self.learning_rate)

	def main(self):
		for i in range(1, self.episodes + 1):
			self.current_episode_buffer = rb.ReplayBuffer(self.steps_in_episode, 32)
			self.uav_model_env.reset_environment()
			print(f"[INFO] Start episode #{i}...", end = '')
			self.start_simulation(i)
			self.uav_model_env.save_episode_p_vectors_to_JSONL(i)
			print("DONE!")
		adf.save_buffer_to_JSONL(self.train_replay_buffer.buffer, f"{gv.REPLAY_BUFFERS_DIR_PATH}/Train_Buffer.jsonl")

	def start_simulation(self, episode_number: int):
		for step_i in range(self.steps_in_episode):
			if self.uav_model_env.early_episode_termination:
				print("[WARN] Early episode termination...")
				break
			train_buffer_size = self.train_replay_buffer.buffer.shape[0]
			if step_i % self.update_actor_critic_NNs_period_in_steps == 0 and train_buffer_size >= self.buffer_batch_size:
				buffer_batch = self.train_replay_buffer.get_batch()
				states, actions, rewards, next_states = self.train_replay_buffer.get_states_actions_rewards_next_states(buffer_batch)
				next_states = tf.cast(next_states, tf.float32)
				self.actor(next_states)
				next_actions = self.actor.get_actions_n_batch()
				log_actions_batch_probabilities = self.actor.get_log_probabilities()
				next_states_actions_batch = self.critic.get_states_actions_batch(next_states, next_actions)
				self.update_critic_Q1_Q2_trainable_weights(
					next_states_actions_batch, rewards, log_actions_batch_probabilities
				)
				self.update_actor_trainable_weights(states, actions)

			current_state = self.uav_model_env.current_state_vector
			action_t = self.actor.get_action(current_state)
			action_t["T_sum"] = (action_t["T_sum"] + 1) * self.uav_model_env.UAV_mass_in_kg * 9.8
			next_state, reward_t = self.uav_model_env.step(action_t)
			transition_vector = self.train_replay_buffer.get_transition_vector(
				current_state, action_t, reward_t, next_state
			)
			self.train_replay_buffer.update_buffer_with_new_transition_vector(transition_vector, step_i)
			self.current_episode_buffer.add_transition_vector_to_buffer(transition_vector, step_i)
		print("=" * 30)
		print(self.current_episode_buffer.buffer)
		adf.save_buffer_to_JSONL(self.current_episode_buffer.buffer, f"{gv.REPLAY_BUFFERS_DIR_PATH}/Buffer_for_Episode-{episode_number}.jsonl")

	def update_critic_Q1_Q2_trainable_weights(self, next_states_actions_batch, rewards_batch, log_actions_batch_probabilities):
		self.critic(next_states_actions_batch)
		print("[INFO]===>Updating Critic Q1, Q2 trainable weights...", end = '')
		with tf.GradientTape() as grad:
			train_vars = [tf.convert_to_tensor(layer_weights) for layer_weights in self.critic.trainable_variables]
			grad.watch(train_vars)

			Q1_batch, Q2_batch = self.critic(next_states_actions_batch)
			min_Q1_Q2 = tf.math.minimum(Q1_batch, Q2_batch)
			rewards = rewards_batch.reshape(len(rewards_batch), 1)
			target_Q = rewards + self.gamma * (min_Q1_Q2 - self.alpha * log_actions_batch_probabilities)
			critic_loss = (Q1_batch - target_Q) ** 2 + (Q2_batch - target_Q) ** 2
			loss_grad = grad.gradient(critic_loss, self.critic.trainable_variables)
			self.critic_optimizer.apply_gradients(zip(loss_grad, self.critic.trainable_variables))
		print("DONE!")

	def update_actor_trainable_weights(self, states_n_batch, actions_n_batch):
		states_actions_batch = self.critic.get_states_actions_batch(states_n_batch, actions_n_batch)
		Q1, Q2 = self.critic(states_actions_batch)
		min_Q1_Q2_batch = tf.math.minimum(Q1, Q2)
		print("[INFO]===>Updating Actor trainable weights...", end = '')
		with tf.GradientTape() as grad:
			train_vars = [tf.convert_to_tensor(layer_weights) for layer_weights in self.actor.trainable_variables]
			grad.watch(train_vars)
			
			means_batch, log_stds_batch = self.actor(states_n_batch)
			log_actions_batch_probabilities = self.actor.get_log_probabilities()
			actor_loss = self.alpha * log_actions_batch_probabilities - min_Q1_Q2_batch
			loss_grad = grad.gradient(actor_loss, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients(zip(loss_grad, self.actor.trainable_variables))
		print("DONE!")

	def create_dirs_for_simulations(self):
		simulations_data_dirs = [gv.SIMULATION_DIR, gv.P_VECTORS_IN_TIME_DIR_PATH, gv.REPLAY_BUFFERS_DIR_PATH]
		for sim_dir in simulations_data_dirs:
			adf.create_dir_if_need(sim_dir)

if __name__ == "__main__":
	UAV_model_environment = uavme.UAV_6DoF_Model_Environment(
		0.99, {"alpha": 2.8, "beta": 0.5, "gamma": 1.2, "delta": 1.4},
		{
			"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
			"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
		},
		{
			"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]}, 
			"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
		},
		0.5, 2, 1000, test_mode = True
	)
	episode_steps = 500
	buffer_batch_size = 32
	fpv_hover_simulation = FPV_Hover_Stabilization_Simulation(
		UAV_model_environment, buffer_batch_size, 0.02, episode_steps, 60
	)
	fpv_hover_simulation.main()
