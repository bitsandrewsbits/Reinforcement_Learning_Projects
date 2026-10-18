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

def get_test_hover_stabilization_simulation_result(algorithm: dict, algorithm_name: str,
test_env: object , target_test_hover_time_in_sec: float,
max_diviation_for_z_stabilization_in_m: float, delta_t_in_sec: float):
	test_episode_hover_steps = int(target_test_hover_time_in_sec / delta_t_in_sec)
	simulation_result = ''
	if algorithm_name == "PID":
		simulation_result = get_PID_algorithm_simulation_result(
			test_env, algorithm["model"], algorithm_name,
			test_episode_steps = test_episode_hover_steps,
			max_diviation_for_stabilization_in_m = max_diviation_for_z_stabilization_in_m
		)
	else:
		simulation_result = get_RL_algorithm_simulation_result(
			test_env, algorithm["trained_model"], algorithm_name,
			test_episode_steps = test_episode_hover_steps,
			max_diviation_for_stabilization_in_m = max_diviation_for_z_stabilization_in_m
		)
	return simulation_result

def get_trained_RL_algorithm(algorithm: dict, algorithm_name: str,
train_env: object, trained_models_path: Path, DR_mode: bool,
parameters_uncertanty: str, external_disturbance_level: str, compute_device: str):
	model_name = get_model_name(
		algorithm_name, DR_mode, parameters_uncertanty, external_disturbance_level
	)
	trained_models_path.mkdir(parents = True, exist_ok = True)
	if f"{model_name}.zip" in os.listdir(trained_models_path):
		print(f"Loading trained {algorithm_name} model...")
		trained_model = algorithm["model"].load(
			path = trained_models_path / f"{model_name}.zip",
			env = train_env,
			device = compute_device
		)
	else:
		trained_model = train_RL_algorithm(
			RL_algorithm = algorithm["model"],
			RL_algorithm_name = algorithm_name,
			total_steps = algorithm["total_steps"],
			save_path = trained_models_path,
			model_name = model_name
		)
	return trained_model

def train_RL_algorithm(
RL_algorithm: object, RL_algorithm_name: str, total_steps: int, save_path: Path, model_name: str) -> object:
	train_start_time = timer()
	RL_algorithm.learn(total_timesteps = total_steps)
	train_end_time = timer()
	total_train_time = train_end_time - train_start_time
	print(f"[INFO] Total train time = {total_train_time} sec.")
	print(f"[INFO] Saving trained model to {save_path}/{model_name}.zip...")
	RL_algorithm.save(save_path / f"{model_name}")
	return RL_algorithm

def get_model_name(algorithm_name: str, DR_mode: bool,
params_uncertanty: str, external_disturb_level: str) -> str:
	# TODO: think, how to separate model names from diff cases.
	if DR_mode:
		model_name = f"{algorithm_name}-DR_6DoF_hover_stab_({params_uncertanty},{external_disturb_level})"
	elif DR_mode == False and external_disturb_level:
		model_name = f"{algorithm_name}-N_6DoF_hover_stab_({external_disturb_level})"
	else:
		model_name = f"{algorithm_name}-N_6DoF_hover_stab"
	print("Model name:", model_name)
	return model_name

def get_RL_algorithm_simulation_result(
env: object, trained_RL_algorithm: object, RL_algorithm_name: str,
test_episode_steps: int, max_diviation_for_stabilization_in_m: float):
	result = {
		"algorithm_name": RL_algorithm_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"hover_time_in_sec": 0.0,
		"hover_steps": 0,
		"stabilization_time_in_sec": 0.0
	}

	observation, info = env.reset()
	current_episode_step = 0
	stabilization_episode_steps = 0
	while current_episode_step < test_episode_steps:
		action, _states = trained_RL_algorithm.predict(observation, deterministic = True)
		observation, reward, terminated, truncated, info = env.step(action)
		
		if terminated or truncated:
			break
		
		if observation[2] < env.hover_stabilization_height_in_m - max_diviation_for_stabilization_in_m:
			stabilization_episode_steps += 1
		elif observation[2] > env.hover_stabilization_height_in_m + max_diviation_for_stabilization_in_m:
			stabilization_episode_steps += 1

		result["observations"].append(observation)
		result["rewards"].append(reward)
		result["terminated"].append(terminated)
		result["truncated"].append(truncated)
		result["hover_time_in_sec"] += 0.01
		result["hover_steps"] += 1
		current_episode_step += 1

	result["stabilization_time_in_sec"] = stabilization_episode_steps * 0.01
	return result

def get_PID_algorithm_simulation_result(
env: object, PID_controller: object, PID_controller_name: str,
test_episode_steps, max_diviation_for_stabilization_in_m: float):
	result = {
		"algorithm_name": PID_controller_name,
		"observations": [], "rewards": [],
		"terminated": [], "truncated": [],
		"hover_time_in_sec": 0.0,
		"hover_steps": 0,
		"stabilization_time_in_sec": 0.0
	}

	observation, info = env.reset()
	PID_controller.reset()
	current_episode_step = 0
	stabilization_episode_steps = 0
	while current_episode_step < test_episode_steps:
		action = PID_controller.main(observation)
		observation, reward, terminated, truncated, info = env.step(action)
		
		if terminated or truncated:
			break
		
		if observation[2] < env.hover_stabilization_height_in_m - max_diviation_for_stabilization_in_m:
			stabilization_episode_steps += 1
		elif observation[2] > env.hover_stabilization_height_in_m + max_diviation_for_stabilization_in_m:
			stabilization_episode_steps += 1

		result["observations"].append(observation)
		result["rewards"].append(reward)
		result["terminated"].append(terminated)
		result["truncated"].append(truncated)
		result["hover_time_in_sec"] += 0.01
		result["hover_steps"] += 1
		current_episode_step += 1

	result["stabilization_time_in_sec"] = stabilization_episode_steps * 0.01
	return result

def get_env_with_random_init_params(DR_mode: bool, target_z: float, 
hover_time_in_sec: float, max_diviation_from_hover_height: float,
reward_coeffs: dict, drag_disturb: bool, wind_disturb: bool,
params_uncertanty: str, wind_gust_level: str):
	test_env = adpt_gym_env.UAV_6DoF_Adapter_For_GymEnv(
		hover_stabilization_height_in_m = target_z,
		hover_time_in_sec = hover_time_in_sec,
		max_diviation_from_hover_height = max_diviation_from_hover_height,
		reward_coefs = reward_coeffs,
		external_drag_disturbance = drag_disturb,
		external_wind_disturbance = wind_disturb,
		simulation_max_steps = 1000,
		DR_mode = DR_mode,
		level_of_params_uncertanty = params_uncertanty,
		wind_gust_disturbance_time_level = wind_gust_level
	)
	return test_env

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
		# print("Z points:", z[:5])
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
	test_episodes_nums = [i for i in range(len(hover_time_values["PID"]))]
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
	hover_time_values = get_algorithms_hover_time_by_test_episodes(all_simulations_results)
	pred_heights_z = get_algorithms_z_by_test_episodes(all_simulations_results)
	stab_times_in_sec = get_algorithms_stabilization_time_by_test_episodes(all_simulations_results)
	algorithm_names = all_simulations_results.keys()
	summary_stats = {}
	for algorithm_name in algorithm_names:
		algorithm_stats = {
			"mean_hover_time_in_sec": 0.0,
			"hover_height_MAE_m": 0.0,
			"hover_height_MAE_STD_m": 0.0,
			"hover_height_RMSE_m": 0.0,
			"hover_height_RMSE_STD_m": 0.0,
			"mean_stabilization_time_in_sec": 0.0,
			"stabilization_time_STD_in_sec": 0.0
		}
		algorithm_stats["mean_hover_time_in_sec"] = round(float(np.average(hover_time_values[algorithm_name])), 2)
		
		MAE, MAE_std = get_MAE_of_test_episodes(pred_heights_z[algorithm_name], target_z)
		algorithm_stats["hover_height_MAE_m"] = round(float(MAE), 4)
		algorithm_stats["hover_height_MAE_STD_m"] = round(float(MAE_std), 4)
		
		RMSE, RMSE_std = get_RMSE_of_test_episodes(pred_heights_z[algorithm_name], target_z)
		algorithm_stats["hover_height_RMSE_m"] = round(float(RMSE), 4)
		algorithm_stats["hover_height_RMSE_STD_m"] = round(float(RMSE_std), 4)

		algorithm_stats["mean_stabilization_time_in_sec"] = round(float(np.average(stab_times_in_sec[algorithm_name])), 2)
		algorithm_stats["stabilization_time_STD_in_sec"] = round(float(
			get_std(stab_times_in_sec[algorithm_name], algorithm_stats["mean_stabilization_time_in_sec"])),
		2)
		print(f"{algorithm_name}: ")
		print(f"Mean_hover_time_in_sec = {algorithm_stats['mean_hover_time_in_sec']} sec.")
		print(f"Hover_heights_MAE_m = {algorithm_stats['hover_height_MAE_m']} +- {algorithm_stats['hover_height_MAE_STD_m']}m")
		print(f"Hover_heights_RMSE_m = {algorithm_stats['hover_height_RMSE_m']} +- {algorithm_stats['hover_height_RMSE_STD_m']} m")
		print(
			(f"mean_stabilization_time_in_sec = {algorithm_stats["mean_stabilization_time_in_sec"]} +-"),
			(f"{algorithm_stats['stabilization_time_STD_in_sec']} sec.")
		)
		print("-" * 40)
		summary_stats[algorithm_name] = algorithm_stats
	return summary_stats

def get_algorithms_hover_time_by_test_episodes(all_simulations_results: dict[list[dict]]) -> dict[list]:
	hover_time_values = {
		algorithm_name: [] for algorithm_name in all_simulations_results.keys()
	}
	for (algorithm_name, simulation_results) in all_simulations_results.items():
		for simulation_result in simulation_results:
			hover_time_values[algorithm_name].append(simulation_result["hover_time_in_sec"])
	return hover_time_values

def get_algorithms_z_by_test_episodes(all_simulations_results: dict[list[dict]]):
	z_values = {
		algorithm_name: [] for algorithm_name in all_simulations_results.keys()
	}
	for (algorithm_name, simulation_results) in all_simulations_results.items():
		for simulation_result in simulation_results:
			episode_z_values = [observation[2] for observation in simulation_result["observations"]]
			z_values[algorithm_name].append(episode_z_values)
	return z_values

def get_algorithms_stabilization_time_by_test_episodes(all_simulations_results: dict[list[dict]]):
	stab_time_in_sec = {
		algorithm_name: [] for algorithm_name in all_simulations_results.keys()
	}
	for (algorithm_name, simulation_results) in all_simulations_results.items():
		for simulation_result in simulation_results:
			stab_time_in_sec[algorithm_name].append(simulation_result["stabilization_time_in_sec"])
	return stab_time_in_sec

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
	PPO_reward_coeffs = {"alpha": 3.0, "beta": 0.4, "gamma": 1.2, "delta": 0.05}
	SAC_reward_coeffs = {"alpha": 3.0, "beta": 0.4, "gamma": 1.2, "delta": 0.05}
	target_train_hover_time_in_sec = 5.0
	target_test_hover_time_in_sec = 10.0
	delta_t_in_sec = 0.01
	test_episode_hover_steps = int(target_test_hover_time_in_sec / delta_t_in_sec)
	max_diviation_from_hover_height = 0.01 # PPO and SAC - are good
	max_diviation_for_z_stabilization_in_m = 0.05
	test_simulations_num = 100
	trained_models_path = Path("trained_models")

	RL_algorithms_train_envs = {
		"PPO": get_env_with_random_init_params(
			DR_mode = False,
			target_z = target_hover_height_in_m,
			hover_time_in_sec = target_test_hover_time_in_sec,
			max_diviation_from_hover_height = max_diviation_from_hover_height,
			reward_coeffs = PPO_reward_coeffs,
			drag_disturb = False,
			wind_disturb = False,
			params_uncertanty = "delta_1",
			wind_gust_level = "zero"
		),
		"SAC": get_env_with_random_init_params(
			DR_mode = False,
			target_z = target_hover_height_in_m,
			hover_time_in_sec = target_test_hover_time_in_sec,
			max_diviation_from_hover_height = max_diviation_from_hover_height,
			reward_coeffs = SAC_reward_coeffs,
			drag_disturb = False,
			wind_disturb = False,
			params_uncertanty = "delta_1",
			wind_gust_level = "zero"
		)
	}
	
	algorithms = {
		"PID": {
			"model": pid6dof.PID_Controller_6DoF(
				target_position_z = target_hover_height_in_m
			)
		},
		"PPO": {
			"model": PPO(
				"MlpPolicy",
				env = RL_algorithms_train_envs["PPO"],
			    learning_rate = 3e-4,
			    n_steps = 2048,
				batch_size = 256,
				n_epochs = 10,
				gamma = 0.99,
				gae_lambda = 0.95,
				verbose = 0,
				device = "cpu"
			),
			"reward_coeffs": {"alpha": 2.5, "beta": 0.4, "gamma": 1.2, "delta": 0.05},
			"total_steps": 1.2e+6,
			"trained_model": ''
		},
		"SAC": {
			"model": SAC(
			    "MlpPolicy",
			    env = RL_algorithms_train_envs["SAC"],
			    learning_rate = 3e-4,
			    batch_size = 256,
			    gamma = 0.99,
			    tau = 0.005,
			    buffer_size = 10 ** 6,
			    verbose = 0,
			    device = "cuda"
			),
			"reward_coeffs": {"alpha": 3.0, "beta": 0.4, "gamma": 1.2, "delta": 0.05},
			"total_steps": 5e+5,
			"trained_model": ''
		}
	}
	algorithms["PPO"]["trained_model"] = get_trained_RL_algorithm(
		algorithm = algorithms["PPO"], algorithm_name = "PPO",
		trained_models_path = trained_models_path,
		train_env = RL_algorithms_train_envs["PPO"],
		DR_mode = False,
		parameters_uncertanty = "delta_1",
		external_disturbance_level = "zero",
		compute_device = "cpu"
	)
	algorithms["SAC"]["trained_model"] = get_trained_RL_algorithm(
		algorithm = algorithms["SAC"], algorithm_name = "SAC",
		trained_models_path = trained_models_path,
		train_env = RL_algorithms_train_envs["SAC"],
		DR_mode = False,
		parameters_uncertanty = "delta_1",
		external_disturbance_level = "zero",
		compute_device = "cuda"
	)

	all_simulations_results = {algorithm_name: [] for algorithm_name in algorithms.keys()}

	print(f"[INFO] Running {test_simulations_num} test simulations for PID, PPO, SAC...")
	for _ in range(test_simulations_num):
		test_env = get_env_with_random_init_params(
			DR_mode = False,
			target_z = target_hover_height_in_m,
			hover_time_in_sec = target_test_hover_time_in_sec,
			max_diviation_from_hover_height = max_diviation_from_hover_height,
			reward_coeffs = SAC_reward_coeffs,
			drag_disturb = False,
			wind_disturb = False,
			params_uncertanty = "delta_1",
			wind_gust_level = "zero"
		)
		for algorithm_name in algorithms.keys():
			simulation_result = get_test_hover_stabilization_simulation_result(
				algorithm = algorithms[algorithm_name],
				algorithm_name = algorithm_name,
				test_env = test_env,
				target_test_hover_time_in_sec = target_test_hover_time_in_sec,
				max_diviation_for_z_stabilization_in_m = max_diviation_for_z_stabilization_in_m
			)
			all_simulations_results[algorithm_name].append(simulation_result)

	result_plots_path = Path("results_plots")
	result_plots_path.mkdir(parents = True, exist_ok = True)
	for i in range(3):
		test_algorithms_result = [
			all_simulations_results["PID"][i],
			all_simulations_results["PPO"][i],
			all_simulations_results["SAC"][i]
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