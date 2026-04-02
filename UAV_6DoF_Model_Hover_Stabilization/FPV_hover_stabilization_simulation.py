# main simulation class with SAC algorithm

import numpy as np
import pandas as pd
import UAV_6DoF_model_environment as uavme
import replay_buffer as rb
import actor_NN as act
import critic_NNs as crit
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# TODO: For now, I see two problems:
# 1) I set hover_time in 2 sec, but in reward function it's not using(!)
# 2) If I will have early episode termination, how to create transition vectors batch if buffer is small for batch?

class FPV_Hover_Stabilization_Simulation:
	def __init__(self, uav_model_env, replay_buffer, actor, critic, learning_rate, steps_in_episode: int, episodes: int):
		self.uav_model_env = uav_model_env
		self.replay_buffer = replay_buffer
		self.actor = actor
		self.critic = critic
		self.steps_in_episode = steps_in_episode
		self.episodes = episodes
		self.update_actor_critic_NNs_period_in_steps = 4
		self.gamma = self.uav_model_env.discount_gamma
		self.alpha = 0.2
		self.learning_rate = learning_rate
		self.critic_optimizer = Adam(self.learning_rate)
		self.actor_optimizer = Adam(self.learning_rate)

	def start_simulations(self):
		for _ in range(self.episodes):
			pass

	def start_simulation(self):
		for step in range(1, self.steps_in_episode + 1):
			# if self.uav_model_env.early_episode_termination:
			# 	print("[WARN] Early episode termination...")
			# 	break
			if step % self.update_actor_critic_NNs_period_in_steps == 0:
				buffer_batch = self.replay_buffer.get_batch()
				states, actions, rewards, next_states = self.replay_buffer.get_states_actions_rewards_next_states(buffer_batch)
				
				states_actions_batch = self.critic.get_states_actions_batch(states, actions)
				next_actions = self.actor(next_states)
				log_actions_batch_probabilities = self.actor.get_log_probabilities()
				self.update_critic_Q1_Q2_trainable_weights(
					states_actions_batch, rewards, log_actions_batch_probabilities
				)
				self.update_actor_trainable_weights(states, actions)

			current_state = self.uav_model_env.current_state_vector
			action_t = self.actor.get_action(current_state)
			next_state, reward_t = self.uav_model_env.step(action_t)
			transition_vector = self.replay_buffer.get_transition_vector(
				current_state, action_t, reward_t, next_state
			)
			self.replay_buffer.add_transition_vector_to_buffer(transition_vector)
		print("Simulation is finished!")
		print("Replay Buffer records transition_vectors:")
		print(self.replay_buffer.buffer)

	def update_critic_Q1_Q2_trainable_weights(self, states_actions_batch, rewards_batch, log_actions_batch_probabilities):
		self.critic(states_actions_batch)
		print("[INFO]===>Updating Critic Q1, Q2 trainable weights...")
		with tf.GradientTape() as grad:
			train_vars = [tf.convert_to_tensor(layer_weights) for layer_weights in self.critic.trainable_variables]
			grad.watch(train_vars)

			Q1_batch, Q2_batch = self.critic(states_actions_batch)
			min_Q1_Q2 = tf.math.minimum(Q1_batch, Q2_batch)
			rewards = rewards_batch.reshape(len(rewards_batch), 1)
			target_Q = rewards + self.gamma * (min_Q1_Q2 - self.alpha * log_actions_batch_probabilities)
			critic_loss = (Q1_batch - target_Q) ** 2 + (Q2_batch - target_Q) ** 2
			loss_grad = grad.gradient(critic_loss, self.critic.trainable_variables)
			self.critic_optimizer.apply(loss_grad, self.critic.trainable_variables)
		print("[INFO] DONE!")

	def update_actor_trainable_weights(self, states_n_batch, actions_n_batch):
		states_actions_batch = self.critic.get_states_actions_batch(states_n_batch, actions_n_batch)
		Q1, Q2 = self.critic(states_actions_batch)
		min_Q1_Q2_batch = tf.math.minimum(Q1, Q2)
		print("[INFO]===>Updating Actor trainable weights...")
		with tf.GradientTape() as grad:
			train_vars = [tf.convert_to_tensor(layer_weights) for layer_weights in self.actor.trainable_variables]
			grad.watch(train_vars)
			
			means_batch, log_stds_batch = self.actor(states_n_batch)
			log_actions_batch_probabilities = self.actor.get_log_probabilities()
			actor_loss = self.alpha * log_actions_batch_probabilities - min_Q1_Q2_batch
			loss_grad = grad.gradient(actor_loss, self.actor.trainable_variables)
			self.actor_optimizer.apply(loss_grad, self.actor.trainable_variables)
		print("[INFO] DONE!")

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
	episode_steps = 10
	replay_buffer = rb.ReplayBuffer(
		pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}),
		episode_steps, 3
	)
	actor = act.Actor(replay_buffer)
	critic = crit.Critic()
	fpv_hover_simulation = FPV_Hover_Stabilization_Simulation(
		UAV_model_environment, replay_buffer, actor, critic, 0.02, episode_steps, 2
	)
	fpv_hover_simulation.start_simulation()
