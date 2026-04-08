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
	dir_path_elems = dir_path.split('/')
	path_levels = len(dir_path_elems)
	if path_levels == 1:
		if dir_path in os.listdir('.'):
			print(f"[INFO] Dir <{dir_path}> already exists!")
		else:
			print(f"[INFO] Creating dir <{dir_path}>...")
			os.mkdir(dir_path)
	else:
		target_dir = dir_path_elems[-1]
		target_search_dir_path = '/'.join(dir_path_elems[:-1])
		if target_dir in os.listdir(target_search_dir_path):
			print(f"[INFO] Dir <{target_dir}> already exists!")
		else:
			print(f"[INFO] Creating dir <{target_dir}>...")
			os.mkdir(dir_path)

if __name__ == "__main__":
	d1 = {"a": [0, 1], "b": [4, 6]}
	d1_values = get_values_as_array(d1)
	print(d1_values)