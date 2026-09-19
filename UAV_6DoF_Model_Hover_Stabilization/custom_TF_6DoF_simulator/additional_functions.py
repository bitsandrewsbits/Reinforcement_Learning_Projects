# additional functions
import numpy as np
import pandas as pd
import os

def get_values_as_array(target_dict: dict[np.array]) -> np.array:
	values = []
	for dict_component_values in target_dict.values():
		values += list(dict_component_values)

	return np.array(values)

def save_dataset_to_JSONL(df: pd.DataFrame, file_path: str):
	df.to_json(file_path, orient = 'records', lines = True)

def get_dataset_from_JSONL(file_path: str):
	return pd.read_json(file_path, orient = 'records', lines = True)

def create_dir_if_need(dir_path):
	dir_path_elems = dir_path.split('/')
	if not os.path.exists(dir_path):
		target_dir = "/".join(dir_path_elems[-2:])
		print(f"[INFO] Creating dir `{target_dir}` with parents dirs...")
		os.makedirs(dir_path)
	else:
		print(f"[INFO] Dir path `{dir_path}` already exists!")

if __name__ == "__main__":
	d1 = {"a": [0, 1], "b": [4, 6]}
	d1_values = get_values_as_array(d1)
	print(d1_values)