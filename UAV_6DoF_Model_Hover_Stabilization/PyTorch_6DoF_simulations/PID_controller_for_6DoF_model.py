# PID controller for 6DoF model
import numpy as np

class PID_Controller_6DoF:
	def __init__(self, target_position_z: float):
		self.target_position_z = target_position_z
		self.delta_t_in_sec = 0.01
		self.g = 9.8

		self.target_position = np.array([0, 0, self.target_position_z])
		self.target_yaw_psi = 0.0

		self.PID_x = PID_Controller(
			K_P = 19.5, K_I = 20.0, K_D = 25.0,
			state_component_name = "x"
		)
		self.PID_y = PID_Controller(
			K_P = 19.5, K_I = 20.0, K_D = 25.0,
			state_component_name = "y"
		)
		self.PID_z = PID_Controller(
			K_P = 40.0, K_I = 25.0, K_D = 35.0,
			state_component_name = "z"
		)

		self.PID_roll_fi = PID_Controller(
			K_P = 20.5, K_I = 3.0, K_D = 5.2,
			state_component_name = "roll_fi"
		)
		self.PID_pitch_teta = PID_Controller(
			K_P = 20.5, K_I = 3.0, K_D = 5.2,
			state_component_name = "pitch_teta"
		)
		self.PID_yaw_psi = PID_Controller(
			K_P = 20.5, K_I = 3.0, K_D = 5.2,
			state_component_name = "yaw_psi"
		)
		self.pid_output_u = np.zeros(4)

	def main(self, current_state: np.array) -> np.array:
		roll_fi, pitch_teta, yaw_psi = self.get_euler_angeles([
			current_state[6], current_state[7],
			current_state[8], current_state[9],
		])

		x = current_state[0]
		y = current_state[1]
		z = current_state[2]

		vx = current_state[3]
		vy = current_state[4]
		vz = current_state[5]

		control_x = self.PID_x.main(self.target_position[0], x, vx)
		control_y = self.PID_y.main(self.target_position[1], y, vy)
		control_z = self.PID_z.main(self.target_position[2], z, vz)

		desired_pitch_teta = np.arctan2(np.sin(control_x / self.g), np.cos(control_x / self.g))
		desired_roll_fi = np.arctan2(np.sin(control_y / self.g), np.cos(control_y / self.g))

		max_angle = np.deg2rad(45.0)

		desired_roll_fi = np.clip(
			desired_roll_fi,
			-max_angle, max_angle
		)

		desired_pitch_teta = np.clip(
			desired_pitch_teta,
			-max_angle, max_angle
		)

		omega_x = current_state[10]
		omega_y = current_state[11]
		omega_z = current_state[12]

		control_roll_fi = self.PID_roll_fi.main(
			desired_roll_fi, roll_fi, omega_x
		)
		control_pitch_teta = self.PID_pitch_teta.main(
			desired_pitch_teta, pitch_teta, omega_y
		)
		control_yaw_psi = self.PID_yaw_psi.main(
			self.target_yaw_psi, yaw_psi, omega_z
		)
		action = self.get_action(
			control_z, control_roll_fi,
			control_pitch_teta, control_yaw_psi
		)
		return action

	def get_action(self, control_z,
	control_roll_fi, control_pitch_teta, control_yaw_psi):
		a1 = (
			control_z - control_pitch_teta + control_yaw_psi
		)
		a2 = (
			control_z + control_roll_fi - control_yaw_psi
		)
		a3 = (
			control_z + control_pitch_teta + control_yaw_psi
		)
		a4 = (
			control_z - control_roll_fi - control_yaw_psi
		)

		actions = np.array([a1, a2, a3, a4])
		action = np.clip(actions, -1.0, 1.0)
		self.pid_output_u = action
		# print("Action:", self.pid_output_u)
		return action

	def reset(self):
		self.PID_x.reset()
		self.PID_y.reset()
		self.PID_z.reset()

		self.PID_roll_fi.reset()
		self.PID_pitch_teta.reset()
		self.PID_yaw_psi.reset()

	def get_euler_angeles(self, q: list) -> np.array:
		qw, qx, qy, qz = q

		sin_roll = 2.0 * (qw * qx + qy * qz)
		cos_roll = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
		roll_fi = np.arctan2(sin_roll, cos_roll)

		sin_pitch = 2.0 * (qw * qy - qz * qx)
		sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
		pitch_teta = np.arcsin(sin_pitch)

		sin_yaw = 2.0 * (qw * qz + qx * qy)
		cos_yaw = 1.0 - 2.0 * (qy ** 2 + qz ** 2)
		yaw_psi = np.arctan2(sin_yaw, cos_yaw)

		return np.array([roll_fi, pitch_teta, yaw_psi])

class PID_Controller:
	def __init__(self, K_P, K_I, K_D, state_component_name: str):
		self.K_P = K_P
		self.K_I = K_I
		self.K_D = K_D
		self.delta_t_in_sec = 0.01
		self.state_component_name = state_component_name

		self.integral_error = 0.0
		self.pid_output_u = 0.0

	def main(self, target_state_component_value, current_state_component_value,
	state_component_derivative) -> np.array:
		proportional_error = self.get_proportional_error(
			target_state_component_value, current_state_component_value
		)
		integral_error = self.get_integral_error(
			target_state_component_value, current_state_component_value
		)
		derivative_error = self.get_derivative_error(state_component_derivative)
		pid_output = 0
		pid_output += self.K_P * proportional_error
		pid_output += self.K_I * integral_error
		pid_output -= self.K_D * derivative_error

		self.pid_output = pid_output
		return pid_output

	def reset(self):
		self.proportional_error = 0.0
		self.integral_error = 0.0
		self.derivative_error = 0.0

	def get_proportional_error(self, target_value, current_value):
		proportional_error = 0.0
		if self.state_component_name == "yaw_psi":
			proportional_angel_error = target_value - current_value
			proportional_angel_error = self.wrap_angle(proportional_angel_error)
			return proportional_angel_error
		else:
			return target_value - current_value

	def get_integral_error(self, target_value, current_value):
		error = self.get_proportional_error(target_value, current_value)
		self.integral_error += error * self.delta_t_in_sec
		return self.integral_error

	def get_derivative_error(self, component_derivative):
		return component_derivative

	def wrap_angle(self, angle):
		return (angle + np.pi) % (2.0 * np.pi) - np.pi