# additional functions
import numpy as np
import pandas as pd
import os

def get_values_as_array(target_dict: dict[np.array]) -> np.array:
	values = []
	for dict_component_values in target_dict.values():
		values += list(dict_component_values)

	return np.array(values)

def save_buffer_to_JSONL(df: pd.DataFrame, file_path: str):
	df.to_json(file_path, orient = 'records', lines = True)

def create_dir_if_need(dir_path):
	if dir_path in os.listdir('.'):
		print(f"[INFO] Dir <{dir_path}> already exists!")
	else:
		print(f"[INFO] Creating dir <{dir_path}>...")
		os.mkdir(dir_path)

if __name__ == "__main__":
	d1 = {"a": [0, 1], "b": [4, 6]}
	d1_values = get_values_as_array(d1)
	print(d1_values)