from modules import data_loader
from modules import k_means

import numpy as np


if __name__ == "__main__":

	images = data_loader.load_data('mnist.pkl.gz')

	init = k_means.k_Means(images)

	dim_list = [None, 2, 5, 10]

	k_list = [2, 3, 5, 10]

	for dim in dim_list:
		for k in k_list:
			init.run(k = k, eigenspace_dim=dim, filtered_list = [3, 9])



