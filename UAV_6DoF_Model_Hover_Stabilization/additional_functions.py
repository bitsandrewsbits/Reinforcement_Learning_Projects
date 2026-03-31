# additional functions
import numpy as np

def get_values_as_array(target_dict: dict[np.array]) -> np.array:
	values = []
	for dict_component_values in target_dict.values():
		values += list(dict_component_values)

	return np.array(values)

if __name__ == "__main__":
	d1 = {"a": [0, 1], "b": [4, 6]}
	d1_values = get_values_as_array(d1)
	print(d1_values)