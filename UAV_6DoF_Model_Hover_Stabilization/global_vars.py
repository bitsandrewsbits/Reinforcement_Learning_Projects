UAV_ACTIONS_SIMULATIONS_DIRS = {
	"ascent": [
		"6DoF_ascent_simulations",
		"6DoF_ascent_simulations/p_vectors_in_time",
		"6DoF_ascent_simulations/replay_buffers"
	],
	"hover_stabilization": [
		"6DoF_hover_stabilization_simulations",
		"6DoF_hover_stabilization_simulations/p_vectors_in_time",
		"6DoF_hover_stabilization_simulations/replay_buffers"
	]
}

P_VECTORS_FILENAME_PATTERN = "p_vectors_for_Episode-"
REPLAY_BUFFERS_FILENAME_PATTERN = "Buffer_for_Episode-"

PASSED_STEPS_BEFORE_LEARNING = 5 * 10 ** 3
SUCCESS_HOVER_HEIGHTS_DATASET = "success_hover_heights_per_episode.jsonl"
SUCCESS_HOVER_HEIGHTS_DATASET_PATH = f"{UAV_ACTIONS_SIMULATIONS_DIRS['hover_stabilization'][1]}/{SUCCESS_HOVER_HEIGHTS_DATASET}"