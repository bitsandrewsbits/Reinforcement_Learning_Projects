# actors training for different actions.
# final simulations - ascent-hover_stabilization
import UAV_6DoF_model_environment as uavme
import FPV_6DoF_action_train_simulation as f6ats
import global_vars as gv
from random import randint
import numpy as np

hover_stabilization_height_in_m = 0.5
hover_stabilization_height_in_sec = 0.5
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
		"p": np.array([0, 0, hover_stabilization_height_in_m + randint(-2, 0) / 100]), "v": np.array([0, 0, 0]),
		"q": np.array([1, 0, 0, 0]), "omega": np.array([0, 0, 0])
	},
	{
		"p": {"x": ["min_x", "max_x"], "y": ["min_y", "max_y"], "z": [0, "max_height"]},
		"q": {"q1": [-0.7, 0.7], "q2": [-0.7, 0.7]}, "T_sum": [0, "max_T_sum"]
	},
	hover_stabilization_height_in_m, hover_stabilization_height_in_sec,
	max_deviation_hover_stabilization_in_m, episode_steps
)

envs_for_actions = [
	UAV_model_environment_ascent_stage,
	UAV_model_environment_hover_stabilization_stage
]

def main(envs_for_actors_actions: list):
	actions_simulations_dirs = gv.UAV_ACTIONS_SIMULATIONS_DIRS.keys()
	for (action_simulations_dirs, env_for_actor_action) in zip(actions_simulations_dirs, envs_for_actors_actions):
		print(f"[INFO] Simulations - UAV {action_simulations_dirs} agent training...")
		fpv_action_actor_training_simulation = f6ats.UAV_6DoF_Action_Simulation(
			action_simulations_dirs, env_for_actor_action, 128, episode_steps, 300
		)
		fpv_action_actor_training_simulation.main()

if __name__ == "__main__":
	main(envs_for_actions)