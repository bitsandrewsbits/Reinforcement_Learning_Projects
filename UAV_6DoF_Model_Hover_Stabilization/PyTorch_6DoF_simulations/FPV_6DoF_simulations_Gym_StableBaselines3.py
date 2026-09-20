# simulations using SAC from Stable-Baseline-v3
import torch
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pathlib import Path

import UAV_6DoF_model_adapter_for_Gymnasium_Env as adpt_gym_env
import PID_controller_for_6DoF_model as pid6dof

def get_RL_algorithm_simulation_results(
env: object, RL_algorithm: object, RL_algorithm_name: str,
train_steps_amount: int, test_episode_steps: int = 200):
	results = {
		"algorithm_name": RL_algorithm_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"time_in_sec": []
	}
	trained_RL_algorithm = train_RL_algorithm(
		RL_algorithm, RL_algorithm_name, train_steps_amount
	)

	observation, info = env.reset()
	current_episode_step = 0
	while current_episode_step < test_episode_steps:
		action, _states = trained_RL_algorithm.predict(observation, deterministic = True)
		observation, reward, terminated, truncated, info = env.step(action)
		
		results["observations"].append(observation)
		results["rewards"].append(reward)
		results["terminated"].append(terminated)
		results["truncated"].append(truncated)
		results["time_in_sec"].append(current_episode_step / 100)
		if terminated or truncated:
			observation, info = env.reset()
		current_episode_step += 1
	return results

def train_RL_algorithm(
RL_algorithm: object, RL_algorithm_name: str, train_steps: int) -> object:
	train_start_time = timer()
	RL_algorithm.learn(total_timesteps = train_steps)
	train_end_time = timer()
	total_train_time = train_end_time - train_start_time
	print(f"[INFO] Total train time = {total_train_time} sec.")
	print(f"[INFO] Saving trained model to {RL_algorithm_name}_6DoF_hover_stabilization.zip...")
	RL_algorithm.save(f"{RL_algorithm_name}_6DoF_hover_stabilization")
	return RL_algorithm

def get_PID_algorithm_simulation_results(
env: object, PID_controller: object, PID_controller_name: str,
test_episode_steps: int = 200):
	results = {
		"algorithm_name": PID_controller_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"time_in_sec": []
	}

	observation, info = env.reset()
	current_episode_step = 0
	while current_episode_step < test_episode_steps:
		action = PID_controller.main(observation)
		observation, reward, terminated, truncated, info = env.step(action)
		
		results["observations"].append(observation)
		results["rewards"].append(reward)
		results["terminated"].append(terminated)
		results["truncated"].append(truncated)
		results["time_in_sec"].append(current_episode_step / 100)
		if terminated or truncated:
			observation, info = env.reset()
		current_episode_step += 1
	return results

def show_x_y_z_by_t(simulation_results: dict):
	fig = plt.figure(figsize = (16.5, 8.5))
	plot_x_by_t_for_algorithms(simulation_results)	
	plot_y_by_t_for_algorithms(simulation_results)
	plot_z_by_t_for_algorithms(simulation_results)
	plt.show()
	return fig

def plot_x_by_t_for_algorithms(simulation_results: list[dict]):
	plt.subplot(3, 1, 1)
	plt.title("x(t)")
	for algorithm_simulation_result in simulation_results:
		x = [observation[0] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			algorithm_simulation_result["time_in_sec"], x,
			label = f"x(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	plt.legend()
	plt.grid()

def plot_y_by_t_for_algorithms(simulation_results: list[dict]):
	plt.subplot(3, 1, 2)
	plt.title("y(t)")
	for algorithm_simulation_result in simulation_results:
		y = [observation[1] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			algorithm_simulation_result["time_in_sec"], y,
			label = f"y(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	plt.legend()
	plt.grid()

def plot_z_by_t_for_algorithms(simulation_results: list[dict]):
	plt.subplot(3, 1, 3)
	plt.title("z(t)")
	for algorithm_simulation_result in simulation_results:
		z = [observation[2] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			algorithm_simulation_result["time_in_sec"], z,
			label = f"z(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	plt.legend()
	plt.grid()

def save_plot_image(figure: object, figures_path: Path):
	figure_path = figures_path / "PID_PPO_SAC_x_y_z_by_t.png"
	figure.savefig(figure_path)

if __name__ == "__main__":
	env = adpt_gym_env.UAV_6DoF_Adapter_For_GymEnv()
	model_SAC = SAC(
	    "MlpPolicy", env,
	    learning_rate = 3 * 10 ** (-4),
	    batch_size = 256, gamma = 0.99,
	    buffer_size = 10 ** 6,
	    verbose = 0, device = "cuda"
	)
	model_PPO = "PPO_obj"
	model_PID = pid6dof.PID_Controller_6DoF()
	
	pid_results = get_PID_algorithm_simulation_results(
		env, model_PID, "PID_Controller",
		test_episode_steps = 1000
	)

	sac_results = get_RL_algorithm_simulation_results(
		env, model_SAC, "SAC_RL_algorithm",
		train_steps_amount = 5 * 10 ** 5,
		test_episode_steps = 1000
	)
	algorithms_results = [pid_results, sac_results]
	
	result_plots_path = Path("results_plots")
	result_plots_path.mkdir(parents = True, exist_ok = True)

	result_fig = show_x_y_z_by_t(algorithms_results)
	save_plot_image(result_fig, result_plots_path)