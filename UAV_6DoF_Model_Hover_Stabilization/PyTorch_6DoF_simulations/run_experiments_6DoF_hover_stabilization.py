# run series of experiments for hover stabilization: 
# with diff params uncertanty levels, external disturbances
import torch
from stable_baselines3 import SAC
from pathlib import Path
import json

import FPV_6DoF_simulations_Gym_StableBaselines3 as f6sgsb3

class SAC_Hover_Stab_Experiments_Manager:
	def __init__(self, trained_models_config: dict,
	experiments_config: dict, experiments_root_path: Path,
	experiment_simulations: int, delta_t_in_sec: float):
		self.trained_models_config = trained_models_config
		self.experiments_config = experiments_config
		self.experiments_root_path = experiments_root_path
		self.trained_models_envs = {}
		self.algorithms = {}
		self.experiment_simulations = experiment_simulations
		self.delta_t_in_sec = delta_t_in_sec

	def main(self):
		self.experiments_root_path.mkdir(parents = True, exist_ok = True)
		self.define_trained_models_envs()
		self.define_algorithms_data_structure()
		self.load_trained_models()
		self.run_experiments()

	def define_trained_models_envs(self):
		for algorithm in self.trained_models_config.keys():
			self.trained_models_envs[algorithm] = f6sgsb3.get_env_with_random_init_params(
				DR_mode = self.trained_models_config[algorithm]["train_env"]["DR_mode"],
				target_z = self.trained_models_config[algorithm]["train_env"]["target_z"],
				hover_time_in_sec = self.trained_models_config[algorithm]["train_env"]["hover_time_in_sec"],
				max_diviation_from_hover_height = (
					self.trained_models_config[algorithm]["train_env"]["max_diviation_from_hover_height"]
				),
				reward_coeffs = self.trained_models_config[algorithm]["train_env"]["reward_coeffs"],
				drag_disturb = self.trained_models_config[algorithm]["train_env"]["drag_disturb"],
				wind_disturb = self.trained_models_config[algorithm]["train_env"]["wind_disturb"],
				params_uncertanty = self.trained_models_config[algorithm]["train_env"]["params_uncertanty"],
				wind_gust_level = self.trained_models_config[algorithm]["train_env"]["wind_gust_level"]
			)

	def define_algorithms_data_structure(self):
		for algorithm in self.trained_models_config.keys():
			self.algorithms[algorithm] = {
				"model": SAC(
				    "MlpPolicy",
				    env = self.trained_models_envs[algorithm],
				    learning_rate = 3e-4,
				    batch_size = 256,
				    gamma = 0.99,
				    tau = 0.005,
				    buffer_size = 10 ** 6,
				    verbose = 0,
				    device = "cuda"
				),
				"trained_model": ''
			}

	def load_trained_models(self):
		for algorithm in self.algorithms.keys():
			self.algorithms[algorithm]["trained_model"] = f6sgsb3.get_trained_RL_algorithm(
				algorithm = self.algorithms[algorithm],
				algorithm_name = self.trained_models_config[algorithm]["load_params"]["algorithm_name"],
				trained_models_path = (
					self.trained_models_config[algorithm]["load_params"]["trained_models_path"]
				),
				train_env = self.trained_models_envs[algorithm],
				DR_mode = self.trained_models_config[algorithm]["train_env"]["DR_mode"],
				parameters_uncertanty = self.trained_models_config[algorithm]["train_env"]["params_uncertanty"],
				external_disturbance_level = self.trained_models_config[algorithm]["train_env"]["wind_gust_level"],
				compute_device = self.trained_models_config[algorithm]["load_params"]["compute_device"]
			)

	def run_experiments(self):
		for experiment_case in self.experiments_config.keys():
			self.get_experiment_result(experiment_case)

	def get_experiment_result(self, experiment_case: str):
		for wind_disturb_level in self.experiments_config[experiment_case]["wind_disturb_level"]:
			for params_uncertanty_level in self.experiments_config[experiment_case]["parameters_uncertanty"]:
				subcase_simulations_results = {}
				for algorithm in self.algorithms.keys():
					subcase_simulations_results[algorithm] = []

				experiment_path = self.create_experiment_dir(params_uncertanty_level, wind_disturb_level)
				print(f"[INFO] Running {self.experiment_simulations} test simulations...")
				for _ in range(self.experiment_simulations):
					test_env = f6sgsb3.get_env_with_random_init_params(
						DR_mode = self.experiments_config[experiment_case]["test_env_params"]["DR_mode"],
						target_z = self.experiments_config[experiment_case]["test_env_params"]["target_z"],
						hover_time_in_sec = self.experiments_config[experiment_case]["test_env_params"]["hover_time_in_sec"],
						max_diviation_from_hover_height = (
							self.experiments_config[experiment_case]["test_env_params"]["max_diviation_from_hover_height"]
						),
						reward_coeffs = self.experiments_config[experiment_case]["test_env_params"]["reward_coeffs"],
						drag_disturb = self.experiments_config[experiment_case]["test_env_params"]["drag_disturb"],
						wind_disturb = self.experiments_config[experiment_case]["test_env_params"]["wind_disturb"],
						params_uncertanty = params_uncertanty_level,
						wind_gust_level = wind_disturb_level
					)

					for algorithm in self.algorithms.keys():
						simulation_result = f6sgsb3.get_test_hover_stabilization_simulation_result(
							algorithm = self.algorithms[algorithm],
							algorithm_name = self.trained_models_config[algorithm]["load_params"]["algorithm_name"],
							test_env = test_env,
							target_test_hover_time_in_sec = (
								self.experiments_config[experiment_case]["test_env_params"]["hover_time_in_sec"]
							),
							max_diviation_for_z_stabilization_in_m = (
								self.experiments_config[experiment_case]["test_env_params"]["max_diviation_from_hover_height"]
							),
							delta_t_in_sec = self.delta_t_in_sec
						)
						subcase_simulations_results[algorithm].append(simulation_result)

				result_plots_path = experiment_path / "test_plots"
				result_plots_path.mkdir(parents = True, exist_ok = True)
				for i in range(1):
					test_experiment_result = [
						subcase_simulations_results[algorithm_name][i] for algorithm_name in self.algorithms.keys()
					]
					target_z = self.trained_models_config["SAC-N"]["train_env"]["target_z"]
					result_fig = f6sgsb3.show_x_y_z_by_t(test_experiment_result, target_z)
					f6sgsb3.save_plot_image(result_fig, result_plots_path, i)
				
				summary_stats = f6sgsb3.get_hover_stabilization_summary_stats(
					subcase_simulations_results, target_hover_height_in_m
				)
				# print(summary_stats)

				subcase_simulations_results["params_uncertanty"] = params_uncertanty_level
				subcase_simulations_results["wind_disturb_level"] = wind_disturb_level
				self.save_experiment_summary_stats(summary_stats, experiment_path)

	def create_experiment_dir(self, params_uncertanty_level: str, wind_disturb_level: str) -> Path:
		print("[INFO] Creating dir for experiment:") 
		print(f"\tparams_uncertanty: {params_uncertanty_level}, wind_disturb_level: {wind_disturb_level}")
		experiment_path = self.experiments_root_path / f"case_{params_uncertanty_level}-{wind_disturb_level}"
		experiment_path.mkdir(parents = True, exist_ok = True)
		return experiment_path

	def save_experiment_summary_stats(self, summary_stats: dict, summary_stats_path: Path):
		summary_stats_file = "summary_stats.json"
		with open(summary_stats_path / summary_stats_file, "w") as cf:
			json.dump(summary_stats, cf, indent = 4)

if __name__ == "__main__":
	target_hover_height_in_m = 1.5
	reward_coeffs = {"alpha": 3.0, "beta": 0.4, "gamma": 1.2, "delta": 0.05}
	target_train_hover_time_in_sec = 5.0
	target_test_hover_time_in_sec = 10.0
	delta_t_in_sec = 0.01
	max_diviation_from_hover_height = 0.01
	max_diviation_for_z_stabilization_in_m = 0.05
	test_simulations_num = 100
	trained_models_path = Path("trained_models")
	compute_device = "cuda" if torch.cuda.is_available() else "cpu"
	
	TRAINED_MODELS_CONFIG = {
		"SAC-N": {
			"train_env": {
				"DR_mode": False,
				"target_z": target_hover_height_in_m,
				"hover_time_in_sec": target_test_hover_time_in_sec,
				"max_diviation_from_hover_height": max_diviation_from_hover_height,
				"reward_coeffs": reward_coeffs,
				"drag_disturb": False,
				"wind_disturb": False,
				"params_uncertanty": "delta_1",
				"wind_gust_level": "zero"
			},
			"load_params": {
				"algorithm_name": "SAC",
				"trained_models_path": trained_models_path,
				"compute_device": compute_device
			}
		},
		"SAC-DR": {
			"train_env": {
				"DR_mode": True,
				"target_z": target_hover_height_in_m,
				"hover_time_in_sec": target_test_hover_time_in_sec,
				"max_diviation_from_hover_height": max_diviation_from_hover_height,
				"reward_coeffs": reward_coeffs,
				"drag_disturb": True,
				"wind_disturb": True,
				"params_uncertanty": "delta_1",
				"wind_gust_level": "low"
			},
			"load_params": {
				"algorithm_name": "SAC",
				"trained_models_path": trained_models_path,
				"compute_device": compute_device
			}
		}
	}
	EXPERIMENTS_CONFIG = {
		"case-1": {
			"test_env_params": {
				"DR_mode": True,
				"target_z": target_hover_height_in_m,
				"hover_time_in_sec": target_test_hover_time_in_sec,
				"max_diviation_from_hover_height": max_diviation_from_hover_height,
				"reward_coeffs": reward_coeffs,
				"drag_disturb": False,
				"wind_disturb": False,
				"params_uncertanty": "",
				"wind_gust_level": ""
			},
			"parameters_uncertanty": ["delta_0", "delta_1", "delta_2", "delta_3"],
			"wind_disturb_level": ["zero"]
		}
	}
	SAC_experiments_manager = SAC_Hover_Stab_Experiments_Manager(
		trained_models_config = TRAINED_MODELS_CONFIG,
		experiments_config = EXPERIMENTS_CONFIG,
		experiments_root_path = Path("6DoF_Hover_Stabilization_Experiments"),
		experiment_simulations = 100,
		delta_t_in_sec = delta_t_in_sec
	)
	SAC_experiments_manager.main()