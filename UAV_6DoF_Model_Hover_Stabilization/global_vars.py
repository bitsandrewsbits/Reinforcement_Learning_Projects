UAV_ACTIONS_SIMULATIONS_DIRS = {
	"ascent": [
		"6DoF_ascent_simulations",
		"6DoF_ascent_simulations/p_vectors_in_time",
		"6DoF_ascent_simulations/replay_buffers",
		"6DoF_ascent_simulations/trained_actor_model",
		"6DoF_ascent_simulations/trained_critics_models",
		"6DoF_ascent_simulations/evaluation_episodes"
	],
	"hover_stabilization": [
		"6DoF_hover_stabilization_simulations",
		"6DoF_hover_stabilization_simulations/p_vectors_in_time",
		"6DoF_hover_stabilization_simulations/replay_buffers",
		"6DoF_hover_stabilization_simulations/trained_actor_model",
		"6DoF_hover_stabilization_simulations/trained_critics_models",
		"6DoF_hover_stabilization_simulations/evaluation_episodes"
	]
}

P_VECTORS_FILENAME_PATTERN = "p_vectors_for_Episode-"
REPLAY_BUFFERS_FILENAME_PATTERN = "Buffer_for_Episode-"

PASSED_STEPS_BEFORE_LEARNING = 5 * 10 ** 3
SUCCESS_HOVER_HEIGHTS_DATASET = "success_hover_heights_per_episode.jsonl"
SUCCESS_HOVER_HEIGHTS_DATASET_PATH = f"{UAV_ACTIONS_SIMULATIONS_DIRS['hover_stabilization'][1]}/{SUCCESS_HOVER_HEIGHTS_DATASET}"

SUCCESS_TRAINED_ACTOR_NN_WEIGHTS = "successfull_actor_NN.weights.h5"
SUCCESS_TRAINED_CRITIC_NNS_WEIGHTS = "successfull_critic_NNs.weights.h5"
SUCCESS_TARGET_CRITIC_NNS_WEIGHTS = "successfull_target_critic_NNs.weights.h5"

RUN_EVALUATION_SIMULATIONS_EVERY_ACTOR_UPDATES = 50
EVALUATION_SIMULATIONS_EPISODE_DIR_PATTERN = "Evaluation_Simulations_after_"