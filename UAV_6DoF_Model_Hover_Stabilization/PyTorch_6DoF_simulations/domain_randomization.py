# domain randomization class for
# generation of vector state, external disturbances
import numpy as np

class Domain_Randomization_For_6DoF:
	def __init__(self, drone_mass: float, level_of_params_uncertanty: str,
	external_wind_disturbance: bool, wind_gust_disturbance_time_level: str,
	episode_steps: int):
		self.drone_mass = drone_mass
		self.F_g = np.array([0, 0, -self.drone_mass * 9.8])
		self.level_of_params_uncertanty = level_of_params_uncertanty
		self.external_wind_disturbance = external_wind_disturbance
		self.wind_gust_disturbance_time_level = wind_gust_disturbance_time_level
		self.episode_steps = episode_steps
		self.levels_6DoF_parameters_uncertanty = {
			"delta_0": 0.0, "delta_1": 0.1,
			"delta_2": 0.2, "delta_3": 0.3
		}
		self.params_uncertanty_value = self.levels_6DoF_parameters_uncertanty[
			self.level_of_params_uncertanty
		]
		self.time_step = 0.01
		self.wind_gust_disturbance_influence_time_levels = {
			"low": 0.5, "medium": 1.0, "strong": 1.5
		}
		self.wind_gust_disturbance_in_steps = int(
			self.wind_gust_disturbance_influence_time_levels[
				self.wind_gust_disturbance_time_level
			] / self.time_step
		)
		self.F_gust_start_point_in_episode = self.get_F_gust_start_point_in_episode()
		self.steps_counter_for_F_gust = 0
		self.previous_wind_velocity = np.zeros(3)
		self.wind_velocity = self.previous_wind_velocity.copy()
		self.wind_update_interval_in_steps = 300
		self.transition_steps_to_update_wind_velocity = 30
		self.current_transition_steps_to_update_wind_velocity = 0
		self.air_density_ro = 1.225
		self.C_D = 1.2
		self.projected_area_A = np.array([0.018, 0.018, 0])

		self.current_episode_step = 0

	def get_F_gust_start_point_in_episode(self):
		return np.random.uniform(0, self.episode_steps // 2)

	def update_current_episode_step(self, episode_step_from_env):
		self.current_episode_step = episode_step_from_env

	def get_random_state_vector_position_components(self, state_vector_component: np.array) -> np.array:
		new_components = []
		for component in state_vector_component:
			new_components.append(self.get_random_state_vector_position_component(component))
		return np.array(new_components)

	def get_random_state_vector_position_component(self, component: float):
		if component == 0.0:
			component = np.random.uniform(-1, 1)
		random_deviation_coefficient = np.random.uniform(
			-self.params_uncertanty_value, self.params_uncertanty_value
		)
		return component * (1 + random_deviation_coefficient)

	def get_random_state_vector_angel_components(self, state_vector_component: np.array) -> np.array:
		new_components = []
		for component in state_vector_component:
			new_components.append(self.get_random_state_vector_angel_component(component))
		return np.array(new_components)

	def get_random_state_vector_angel_component(self, component: float):
		if component == 0.0:
			component = np.random.uniform(-45, 45)
		random_deviation_coefficient = np.random.uniform(
			-self.params_uncertanty_value, self.params_uncertanty_value
		)
		return component * (1 + random_deviation_coefficient)

	def get_random_drone_mass(self):
		return self.drone_mass * np.random.uniform(0.95, 1.05)

	def get_random_drag(self, drag: np.array):
		return drag * np.random.uniform(0.9, 1.1)

	def get_F_wind_gust(self, current_state_speed_component: np.array):
		if self.external_wind_disturbance:
			F_wind = self.get_F_wind(current_state_speed_component)
			if self.current_episode_step == self.F_gust_start_point_in_episode:
				F_gust = self.get_F_gust()
			F_wind_gust = F_wind + F_gust
			return F_wind_gust
		else:
			return 0.0

	def get_F_gust(self):
		if self.steps_counter_for_F_gust >= 0 and self.steps_counter_for_F_gust < self.wind_gust_disturbance_in_steps:
			F_gust = self.F_g
			self.steps_counter_for_F_gust += 1
		else:
			self.steps_counter_for_F_gust = 0
			F_gust = 0.0
		return F_gust

	def get_F_wind(self, current_state_speed_component: np.array):
		self.define_current_step_wind_velocity()
		v_relative = self.wind_velocity - current_state_speed_component
		F_wind = 0.5 * self.air_density_ro * (self.C_D * self.projected_area_A) * abs(v_relative) * v_relative
		return F_wind

	def define_current_step_wind_velocity(self):
		if self.current_step % self.wind_update_interval_in_steps == 0:
			self.generate_wind_velocity()
		else:
			self.update_wind_velocity_to_new_via_transition()

	def generate_wind_velocity(self):
		self.previous_wind_velocity = self.wind_velocity.copy()
		self.wind_velocity = np.array([
			self.RNG.uniform(-2, 2),
			self.RNG.uniform(-2, 2),
			0.0
		])
		self.current_steps_to_update_wind_velocity = 0

	def update_wind_velocity_to_new_via_transition(self):
		alpha = self.get_wind_velocity_transition_coeff_alpha()
		if alpha <= 1:
			self.wind_velocity = (1 - alpha) * self.previous_wind_velocity + alpha * self.wind_velocity
		else:
			self.current_transitions_steps_to_update_wind_velocity = 0

	def get_wind_velocity_transition_coeff_alpha(self):
		alpha = round(self.current_transition_steps_to_update_wind_velocity / self.transition_steps_to_update_wind_velocity, 3)
		self.current_transition_steps_to_update_wind_velocity += 1
		return alpha