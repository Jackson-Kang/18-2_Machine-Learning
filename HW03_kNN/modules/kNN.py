import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import time

class knn_init:

	'''
	
	kNN definition class

		k
		   - initial value of 'k'
		   - default value: 1 ( Nearest Neighbor )

		data (mnist)
		   - dim: (train_x, train_y, test_x, test_y)

	'''

	def __init__(self, data, k_value=1):
		self._k = k_value
		self._train_x, self._train_y, self._test_x, self._test_y = data

	def run(self):

		accuracy = 0.0
		test_number = 10000

		print("Start clssification... ")

		start_time = time.time()
		for test_img_idx in np.ndindex(self._test_x.shape[:1]):
			dist_list = np.array([])

			#if test_img_idx[0] % 1000:
			#	print("\ttest image " + str(test_img_idx[0]+1) + "th")

			#for train_img_idx in np.ndindex(self._train_x.shape[:1]):
			distance = np.linalg.norm(self._test_x[test_img_idx] - self._train_x, axis=1)	# get euclidean distance
			sorted_indices = np.argsort(distance)	# sorted by order of min dist

			pred_class = stats.mode(self._train_y[sorted_indices[:self._k]])[0][0]
			#print("\t\tpred: ", pred_class, "real: ", self._test_y[test_img_idx], pred_class == self._test_y[test_img_idx])

			if self._test_y[test_img_idx] ==pred_class:
				accuracy += 1

			if test_img_idx[0] == test_number:
				break


		end_time = time.time()

		print("\n[Result] Total Acc is ", accuracy/test_number, ", ", end_time - start_time, "seconds")
		
