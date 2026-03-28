# main simulation class with SAC algorithm

import numpy as np
import pandas as pd
import UAV_6DoF_model_environment as uavme
import replay_buffer as rb
import actor_NN as act
import critic_NNs as crit

class FPV_Hover_Stabilization_Simulation:
	def __init__(self, uav_model_env, replay_buffer, actor, critic, steps_in_episode: int, episodes: int):
		self.uav_model_env = uav_model_env
		self.replay_buffer = replay_buffer
		self.actor = actor
		self.critic = critic
		self.steps_in_episode = steps_in_episode
		self.episodes = episodes
		self.update_actor_critic_NNs_period_in_steps = 10
		self.gamma = self.uav_model_env.discount_gamma
		self.alpha = 0.2

	def start_simulations(self):
		for _ in range(self.episodes):
			pass

	# TODO: think, how to fill replay buffer with transition vectors without termination.
	# TODO: create critic simple networks
	# TODO: integrate critic networks to main simulation class for actor, critic networks updating via
	# train-loop via transition_vectors batches.

	def start_simulation(self):
		for step in range(1, self.steps_in_episode + 1):
			# if self.uav_model_env.early_episode_termination:
			# 	print("[WARN] Early episode termination...")
			# 	break
			if step % self.update_actor_critic_NNs_period_in_steps == 0:
				print("[INFO] Updating actor, critic NNs...")
				buffer_batch = self.replay_buffer.get_batch()
				states, actions, rewards, next_states = self.replay_buffer.get_states_actions_rewards_next_states(buffer_batch)
				next_actions = self.actor.get_actions_n_batch(next_states)
				
				target_Q = self.critic.get_target_Q(rewards, self.gamma, self.alpha, next_actions)
				critic_loss = self.critic.get_critic_loss(states, actions, target_Q)

				actions = self.actor.get_actions_n_batch(states)
				states_actions_batch = self.critic.get_states_actions_batch(states, actions)
				self.critic.define_min_Q1_Q2_batch(states_actions_batch)
				actor_loss = self.alpha * tf.math.log(actions) - self.critic.min_Q1_Q2_batch

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
	replay_buffer = rb.ReplayBuffer(
		pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}),
		25, 3
	)
	actor = act.Actor(replay_buffer)
	critic = crit.Critic()
	fpv_hover_simulation = FPV_Hover_Stabilization_Simulation(
		UAV_model_environment, replay_buffer, actor, critic, 10, 2
	)
	fpv_hover_simulation.start_simulation()
