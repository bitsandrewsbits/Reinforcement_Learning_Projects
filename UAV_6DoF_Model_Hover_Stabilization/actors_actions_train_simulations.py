# actors training for different actions.
# final simulations - ascent-hover_stabilization
import UAV_6DoF_model_environment as uavme
import FPV_6DoF_action_train_simulation as f6ats
import global_vars as gv
from random import randint
import numpy as np
import os

hover_stabilization_height_in_m = 0.5
hover_stabilization_height_in_sec = 1
max_deviation_hover_stabilization_in_m = 0.02
episode_steps = 1000

UAV_model_environment_ascent_stage = uavme.UAV_6DoF_Model_Environment(
	"ascent", 0.99, {"alpha_xy": 0.5, "alpha_z": 15, "beta": 10, "gamma": 0.05, "delta": 10 ** (-5)},
	{
		"p": np.array([0, 0, 0]), "v": np.array([0, 0, 0]),
		"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
	},
	{
		"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]},
		"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
	},
	hover_stabilization_height_in_m, hover_stabilization_height_in_sec,
	max_deviation_hover_stabilization_in_m, episode_steps
)

UAV_model_environment_hover_stabilization_stage = uavme.UAV_6DoF_Model_Environment(
	"hover_stabilization", 0.99, {"alpha_xy": 0.5, "alpha_z": 50, "beta": 10, "gamma": 0.5, "delta": 10 ** (-5)},
	{
		"p": np.array([0, 0, hover_stabilization_height_in_m - max_deviation_hover_stabilization_in_m]), "v": np.array([0, 0, 0]),
		"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
	},
	{
		"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]},
		"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
	},
	hover_stabilization_height_in_m, hover_stabilization_height_in_sec,
	max_deviation_hover_stabilization_in_m, episode_steps
)

envs_for_actions = {
	"ascent": UAV_model_environment_ascent_stage,
	"hover_stabilization": UAV_model_environment_hover_stabilization_stage
}

def main(envs_for_actors_actions: dict):
	for (action_name, action_env) in envs_for_actors_actions.items():
		print(f"[INFO] Simulations - UAV {action_name} agent training...")
		delete_old_p_vectors_replay_buffers(action_name)
		fpv_action_actor_training_simulation = f6ats.UAV_6DoF_Action_Simulation(
			action_name, action_env, 128, episode_steps, 500
		)
		fpv_action_actor_training_simulation.main()

def delete_old_p_vectors_replay_buffers(uav_action_dirs: str):
	uav_action_p_vectors_path = gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action_dirs][1]
	uav_action_replay_buffers_path = gv.UAV_ACTIONS_SIMULATIONS_DIRS[uav_action_dirs][2]
	if os.path.exists(uav_action_p_vectors_path):
		print("[WARN] UAV Action p-vectors logs already exist! Removing...")
		os.system(f"rm -r {uav_action_p_vectors_path}")
	if os.path.exists(uav_action_replay_buffers_path):
		print("[WARN] UAV Action replay buffers logs already exist! Removing...")
		os.system(f"rm -r {uav_action_replay_buffers_path}")

if __name__ == "__main__":
	main(envs_for_actions)