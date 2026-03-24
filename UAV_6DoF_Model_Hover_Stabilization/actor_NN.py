# Actor NN for SAC
# state_t -> NN -> action_t
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import replay_buffer as rb

def test_actor_NN(replay_buffer):
	for _ in range(2):
		print("Batch:")
		batch = replay_buffer.get_batch()
		states_n_batch = get_states_n_batch(batch)
		mean_std_batch = get_mean_log_std_batches(states_n_batch)
		print("mean-log_std batch:", mean_std_batch)

def get_actions_n_batch(mean_std_batch):
	# TODO: think, how to do actions_n batch = means + stds * noise, noise - N(0, 1).
	pass

def get_mean_log_std_batches(states_n_batch: np.array):
	dense_layer_1 = Dense(256, activation = 'relu')(states_n_batch)
	dense_layer_2 = Dense(256, activation = 'relu')(dense_layer_1)
	mean_batch = Dense(4)(dense_layer_2)
	log_std_batch = Dense(4)(dense_layer_2)
	
	return mean_batch, log_std_batch

def get_states_n_batch(batch: pd.DataFrame):
	return np.array(list(batch["state_t"].values))

if __name__ == "__main__":
	buffer = rb.ReplayBuffer(
		pd.DataFrame({"state_t": [], "action_t": [], "reward_t": [], "state_tp1": []}),
		10, 3
	)
	buffer.test_buffer_on_random_vectors()
	test_actor_NN(buffer)