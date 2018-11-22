import numpy as np

def load_cis_data(path_list, index):

	return _parse_data(path_list[index])
	


def _parse_data(path):

	with open(path, "r", encoding="utf-8") as f:
		x_list, labels = [], []

		for line in f.readlines():
			data = line.split("\t")
			x_list.append([float(data[0]), float(data[1])])
			labels.append(int(data[2][:-1]))

	return np.asarray(x_list), np.asarray(labels)
