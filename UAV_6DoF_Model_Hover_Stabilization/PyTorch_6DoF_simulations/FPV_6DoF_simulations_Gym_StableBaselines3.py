# simulations using SAC from Stable-Baseline-v3
import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO
import numpy as np
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pathlib import Path
import os

import UAV_6DoF_model_adapter_for_Gymnasium_Env as adpt_gym_env
import PID_controller_for_6DoF_model as pid6dof

def train_RL_algorithm(
RL_algorithm: object, RL_algorithm_name: str, total_steps: int, save_path: Path) -> object:
	train_start_time = timer()
	RL_algorithm.learn(total_timesteps = total_steps)
	train_end_time = timer()
	total_train_time = train_end_time - train_start_time
	print(f"[INFO] Total train time = {total_train_time} sec.")
	print(f"[INFO] Saving trained model to {save_path}/{RL_algorithm_name}_6DoF_hover_stabilization.zip...")
	RL_algorithm.save(save_path / f"{RL_algorithm_name}_6DoF_hover_stabilization")
	return RL_algorithm

def get_RL_algorithm_simulation_result(
env: object, trained_RL_algorithm: object, RL_algorithm_name: str,
test_episode_steps: int = 200):
	result = {
		"algorithm_name": RL_algorithm_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"hover_time_in_sec": 0.0,
		"hover_steps": 0
	}

	observation, info = env.reset()
	current_episode_step = 0
	while current_episode_step < test_episode_steps:
		action, _states = trained_RL_algorithm.predict(observation, deterministic = True)
		observation, reward, terminated, truncated, info = env.step(action)
		
		if terminated or truncated:
			break
		result["observations"].append(observation)
		result["rewards"].append(reward)
		result["terminated"].append(terminated)
		result["truncated"].append(truncated)
		result["hover_time_in_sec"] += 0.01
		result["hover_steps"] += 1
		current_episode_step += 1
	return result

def get_PID_algorithm_simulation_result(
env: object, PID_controller: object, PID_controller_name: str,
test_episode_steps: int = 200, hover_time_in_steps: int = 100):
	result = {
		"algorithm_name": PID_controller_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"hover_time_in_sec": 0.0,
		"hover_steps": 0
	}

	observation, info = env.reset()
	PID_controller.reset()
	current_episode_step = 0
	while current_episode_step < test_episode_steps:
		action = PID_controller.main(observation)
		observation, reward, terminated, truncated, info = env.step(action)
		
		if terminated or truncated:
			break
		result["observations"].append(observation)
		result["rewards"].append(reward)
		result["terminated"].append(terminated)
		result["truncated"].append(truncated)
		result["hover_time_in_sec"] += 0.01
		result["hover_steps"] += 1
		current_episode_step += 1
	return result

def get_test_env_with_random_init_params(DR_mode, target_z, hover_time_in_sec = 5.0):
	test_env = adpt_gym_env.UAV_6DoF_Adapter_For_GymEnv(
		hover_stabilization_height_in_m = target_z,
		hover_time_in_sec = hover_time_in_sec,
		external_drag_disturbance = True,
		external_wind_disturbance = True,
		simulation_max_steps = 1000,
		DR_mode = DR_mode,
		level_of_params_uncertanty = "delta_1",
		wind_gust_disturbance_time_level = "low"
	)
	return test_env

def get_train_env_with_random_init_params(DR_mode, target_z, hover_time_in_sec = 5.0):
	train_env = adpt_gym_env.UAV_6DoF_Adapter_For_GymEnv(
		hover_stabilization_height_in_m = target_z,
		hover_time_in_sec = hover_time_in_sec,
		external_drag_disturbance = True,
		external_wind_disturbance = True,
		simulation_max_steps = 1000,
		DR_mode = DR_mode,
		level_of_params_uncertanty = "delta_1",
		wind_gust_disturbance_time_level = "low"
	)
	return train_env

def show_x_y_z_by_t(simulation_results: list[dict], target_z: float):
	fig = plt.figure(figsize = (16.5, 8.5))
	plot_x_by_t_for_algorithms(simulation_results)
	plot_y_by_t_for_algorithms(simulation_results)
	plot_z_by_t_for_algorithms(simulation_results, target_z)
	plt.xlabel("time, sec")
	plt.ylabel("meters, m")
	plt.show()
	return fig

def plot_x_by_t_for_algorithms(simulation_results: list[dict]):
	plt.subplot(3, 1, 1)
	plt.title("x(t)")
	for algorithm_simulation_result in simulation_results:
		time_in_sec = np.linspace(
			0, algorithm_simulation_result["hover_time_in_sec"],
			algorithm_simulation_result["hover_steps"]
		)
		x = [observation[0] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			time_in_sec, x,
			label = f"x(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	plt.legend()
	plt.grid()

def plot_y_by_t_for_algorithms(simulation_results: list[dict]):
	plt.subplot(3, 1, 2)
	plt.title("y(t)")
	for algorithm_simulation_result in simulation_results:
		time_in_sec = np.linspace(
			0, algorithm_simulation_result["hover_time_in_sec"],
			algorithm_simulation_result["hover_steps"]
		)
		y = [observation[1] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			time_in_sec, y,
			label = f"y(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	plt.legend()
	plt.grid()

def plot_z_by_t_for_algorithms(simulation_results: list[dict], target_z: float):
	plt.subplot(3, 1, 3)
	plt.title("z(t)")
	for algorithm_simulation_result in simulation_results:
		time_in_sec = np.linspace(
			0, algorithm_simulation_result["hover_time_in_sec"],
			algorithm_simulation_result["hover_steps"]
		)
		z = [observation[2] for observation in algorithm_simulation_result["observations"]]
		plt.plot(
			time_in_sec, z,
			label = f"z(t)-{algorithm_simulation_result["algorithm_name"]}"
		)
	hover_line_points = [target_z for _ in range(len(simulation_results[0]["observations"]))]
	time_in_sec = np.linspace(
		0, simulation_results[0]["hover_time_in_sec"],
		simulation_results[0]["hover_steps"]
	)
	plt.plot(time_in_sec, hover_line_points, label = "Hover-target-z")
	plt.legend()
	plt.grid()

def show_stabilization_statistics_by_algorithms(all_simulations_results: dict[list[dict]]):
	fig = plt.figure(figsize = (16.5, 8.5))

	hover_time_values = get_algorithms_hover_time_by_test_episodes(all_simulations_results)
	test_episodes_nums = [i for i in range(len(hover_time_values["PID_Controller"]))]
	plt.title("Move-time in save zone by time (PID, PPO, SAC)")
	for algorithm_name, hover_times_in_sec in hover_time_values.items():
		plt.scatter(
			test_episodes_nums, hover_times_in_sec,
			marker = '.', label = f"hover_time_({algorithm_name})"
		)
	plt.xlabel("Episode time, sec")
	plt.ylabel("Time in save zone, sec")
	plt.grid()
	plt.legend()
	plt.show()
	return fig

def save_plot_image(figure: object, figures_path: Path, test_episode_num: int = 0, plot_name = None):
	if plot_name:
		figure_path = figures_path / f"{plot_name}.png"
	else:
		figure_path = figures_path / f"PID_PPO_SAC_x_y_z_by_t_Test_Episode_#{test_episode_num}.png"
	figure.savefig(figure_path)

def get_hover_stabilization_summary_stats(all_simulations_results: dict[list[dict]], target_z: float):
	summary_stats = {
		"PID_Controller": {
			"mean_hover_time_in_sec": 0.0,
			"hover_height_MAE_m": 0.0,
			"hover_height_MAE_STD_m": 0.0,
			"hover_height_RMSE_m": 0.0,
			"hover_height_RMSE_STD_m": 0.0
		},
		"PPO_RL_algorithm": {
			"mean_hover_time_in_sec": 0.0,
			"hover_height_MAE_m": 0.0,
			"hover_height_MAE_STD_m": 0.0,
			"hover_height_RMSE_m": 0.0,
			"hover_height_RMSE_STD_m": 0.0
		},
		"SAC_RL_algorithm": {
			"mean_hover_time_in_sec": 0.0,
			"hover_height_MAE_m": 0.0,
			"hover_height_MAE_STD_m": 0.0,
			"hover_height_RMSE_m": 0.0,
			"hover_height_RMSE_STD_m": 0.0
		}
	}
	hover_time_values = get_algorithms_hover_time_by_test_episodes(all_simulations_results)
	for algorithm_name, hover_time in hover_time_values.items():
		pred_heights_z = get_algorithms_z_by_test_episodes(all_simulations_results)
		summary_stats[algorithm_name]["mean_hover_time_in_sec"] = round(np.average(hover_time), 2)
		MAE, MAE_std = get_MAE_of_test_episodes(pred_heights_z[algorithm_name], target_z)
		summary_stats[algorithm_name]["hover_height_MAE_m"] = MAE
		summary_stats[algorithm_name]["hover_height_MAE_STD_m"] = MAE_std
		RMSE, RMSE_std = get_RMSE_of_test_episodes(pred_heights_z[algorithm_name], target_z)
		summary_stats[algorithm_name]["hover_height_RMSE_m"] = RMSE
		summary_stats[algorithm_name]["hover_height_RMSE_STD_m"] = RMSE_std
		print(f"{algorithm_name}: ")
		print(f"Mean_hover_time_in_sec = {summary_stats[algorithm_name]['mean_hover_time_in_sec']} sec")
		print(f"Hover_heights_MAE_m = {summary_stats[algorithm_name]['hover_height_MAE_m']} +- {summary_stats[algorithm_name]['hover_height_MAE_STD_m']}m")
		print(f"Hover_heights_RMSE_m = {summary_stats[algorithm_name]['hover_height_RMSE_m']} +- {summary_stats[algorithm_name]['hover_height_RMSE_STD_m']} m")
		print("-" * 40)
	return summary_stats

def get_algorithms_hover_time_by_test_episodes(all_simulations_results: dict[list[dict]]) -> dict[list]:
	hover_time_values = {
		"PID_Controller": [],
		"PPO_RL_algorithm": [],
		"SAC_RL_algorithm": []
	}
	for (algorithm_name, simulation_results) in all_simulations_results.items():
		for simulation_result in simulation_results:
			hover_time_values[algorithm_name].append(simulation_result["hover_time_in_sec"])
	return hover_time_values

def get_algorithms_z_by_test_episodes(all_simulations_results: dict[list[dict]]):
	z_values = {
		"PID_Controller": [],
		"PPO_RL_algorithm": [],
		"SAC_RL_algorithm": []
	}
	for (algorithm_name, simulation_results) in all_simulations_results.items():
		for simulation_result in simulation_results:
			z_values[algorithm_name].append(simulation_result["observations"][2])
	return z_values

def get_MAE_of_test_episodes(z_by_episodes: list[np.array], target_z: float):
	episodes_averages = []
	for episode_z in z_by_episodes:
		episodes_averages.append(get_MAE_z_of_episode(episode_z, target_z))
	aver_of_aver = get_average_of_averages_z_of_episodes(episodes_averages)
	MAE_std = get_std(episodes_averages, aver_of_aver)
	return aver_of_aver, MAE_std

def get_RMSE_of_test_episodes(z_by_episodes: list[np.array], target_z: float):
	episodes_averages = []
	for episode_z in z_by_episodes:
		episodes_averages.append(get_RMSE_z_of_episode(episode_z, target_z))
	aver_of_aver = get_average_of_averages_z_of_episodes(episodes_averages)
	RMSE_std = get_std(episodes_averages, aver_of_aver)
	return aver_of_aver, RMSE_std

def get_std(episodes_averages, aver_of_aver: float):
	return np.sqrt(
		(1 / (len(episodes_averages) - 1)) * np.sum(
			[(episode_aver - aver_of_aver) ** 2 for episode_aver in episodes_averages]
		)
	)

def get_average_of_averages_z_of_episodes(episodes_averages: np.array):
	return np.average(episodes_averages)

def get_MAE_z_of_episode(pred_heights_z: np.array, target_z: float):
	return round(np.average(
		[abs(target_z - predicted_z) for predicted_z in pred_heights_z]
	), 3)

def get_RMSE_z_of_episode(pred_heights_z: np.array, target_z: float):
	return round(np.sqrt(np.mean(
		[abs(predicted_z - target_z) ** 2 for predicted_z in pred_heights_z]
	)), 3)

def show_MAE_comparison_plot(summary_stats: dict[dict]):
	fig = plt.figure(figsize = (5, 5.5))
	plt.title("height-z MAE")
	mae = [alg_summary["hover_height_MAE_m"] for alg_summary in summary_stats.values()]
	mae_err = [alg_summary["hover_height_MAE_STD_m"] for alg_summary in summary_stats.values()]
	plt.bar(
		x = list(summary_stats.keys()), height = mae, 
		width = [0.3, 0.3, 0.3], yerr = mae_err, 
		color = 'skyblue', edgecolor = 'black', capsize = 4)
	plt.show()
	return fig

def show_RMSE_comparison_plot(summary_stats: dict[dict]):
	fig = plt.figure(figsize = (5, 5.5))
	plt.title("height-z RMSE")
	mae = [alg_summary["hover_height_RMSE_m"] for alg_summary in summary_stats.values()]
	mae_err = [alg_summary["hover_height_RMSE_STD_m"] for alg_summary in summary_stats.values()]
	plt.bar(
		x = list(summary_stats.keys()), height = mae, 
		width = [0.3, 0.3, 0.3], yerr = mae_err, 
		color = 'skyblue', edgecolor = 'black', capsize = 4)
	plt.show()
	return fig
 
if __name__ == "__main__":
	target_hover_height_in_m = 1.5
	target_train_hover_time_in_sec = 5.0
	target_test_hover_time_in_sec = 10.0
	delta_t_in_sec = 0.01
	# train_episode_hover_steps = int(target_train_hover_time_in_sec / delta_t_in_sec)
	test_episode_hover_steps = int(target_test_hover_time_in_sec / delta_t_in_sec)
	
	train_env = get_train_env_with_random_init_params(
		DR_mode = True,
		target_z = target_hover_height_in_m,
		hover_time_in_sec = target_train_hover_time_in_sec
	)

	model_PID = pid6dof.PID_Controller_6DoF(
		target_position_z = target_hover_height_in_m
	)

	trained_models_path = Path("trained_models")
	trained_models_path.mkdir(parents = True, exist_ok = True)

	if "PPO_RL_algorithm_6DoF_hover_stabilization.zip" in os.listdir(trained_models_path):
		print("Loading trained PPO model...")
		trained_PPO = PPO.load(
			path = trained_models_path / "PPO_RL_algorithm_6DoF_hover_stabilization.zip",
			env = train_env,
			device = "cuda"
		)
	else:
		model_PPO = PPO(
			"MlpPolicy",
			env = train_env,
		    learning_rate = 3e-4,
		    n_steps = 2048,
			batch_size = 256,
			n_epochs = 10,
			gamma = 0.99,
			gae_lambda = 0.95,
			verbose = 0,
			device = "cpu"
		)
		trained_PPO = train_RL_algorithm(
			RL_algorithm = model_PPO,
			RL_algorithm_name = "PPO_RL_algorithm",
			total_steps = 1e+6,
			save_path = trained_models_path
		)
	if "SAC_RL_algorithm_6DoF_hover_stabilization.zip" in os.listdir(trained_models_path):
		print("Loading trained SAC model...")
		trained_SAC = SAC.load(
			path = trained_models_path / "SAC_RL_algorithm_6DoF_hover_stabilization.zip",
			env = train_env,
			device = "cuda"
		)
	else:
		model_SAC = SAC(
		    "MlpPolicy",
		    env = train_env,
		    learning_rate = 3e-4,
		    batch_size = 256,
		    gamma = 0.99,
		    tau = 0.005,
		    buffer_size = 10 ** 6,
		    verbose = 0,
		    device = "cuda"
		)
		trained_SAC = train_RL_algorithm(
			RL_algorithm = model_SAC,
			RL_algorithm_name = "SAC_RL_algorithm",
			total_steps = 5e+5,
			save_path = trained_models_path
		)

	all_simulations_results = {
		"PID_Controller": [],
		"PPO_RL_algorithm": [],
		"SAC_RL_algorithm": []
	}
	print("[INFO] Running 100 test simulations for PID, PPO, SAC...")
	test_simulation_num = 100
	for _ in range(test_simulation_num):
		test_env = get_test_env_with_random_init_params(
			DR_mode = True,
			target_z = target_hover_height_in_m,
			hover_time_in_sec = target_test_hover_time_in_sec
		)

		pid_result = get_PID_algorithm_simulation_result(
			test_env, model_PID, "PID_Controller",
			test_episode_steps = test_episode_hover_steps
		)
		ppo_result = get_RL_algorithm_simulation_result(
			test_env, trained_PPO, "PPO_RL_algorithm",
			test_episode_steps = test_episode_hover_steps
		)
		sac_result = get_RL_algorithm_simulation_result(
			test_env, trained_SAC, "SAC_RL_algorithm",
			test_episode_steps = test_episode_hover_steps
		)
		all_simulations_results["PID_Controller"].append(pid_result)
		all_simulations_results["PPO_RL_algorithm"].append(ppo_result)
		all_simulations_results["SAC_RL_algorithm"].append(sac_result)

	result_plots_path = Path("results_plots")
	result_plots_path.mkdir(parents = True, exist_ok = True)
	for i in range(3):
		test_algorithms_result = [
			all_simulations_results["PID_Controller"][i],
			all_simulations_results["PPO_RL_algorithm"][i],
			all_simulations_results["SAC_RL_algorithm"][i]
		]
		result_fig = show_x_y_z_by_t(test_algorithms_result, target_hover_height_in_m)
		save_plot_image(result_fig, result_plots_path, i)

	all_test_stabs_fig = show_stabilization_statistics_by_algorithms(all_simulations_results)
	save_plot_image(all_test_stabs_fig, result_plots_path, 0, "All_test_episodes_stabilization_task")
	
	summary_stats = get_hover_stabilization_summary_stats(all_simulations_results, target_hover_height_in_m)
	MAE_comparison_plot = show_MAE_comparison_plot(summary_stats)
	save_plot_image(MAE_comparison_plot, result_plots_path, 0, "height_MAE_by_algorithms")
	RMSE_comparison_plot = show_RMSE_comparison_plot(summary_stats)
	save_plot_image(RMSE_comparison_plot, result_plots_path, 0, "height_RMSE_by_algorithms")