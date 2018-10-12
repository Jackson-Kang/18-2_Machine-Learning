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

	def __init__(self, data):
		self._train_x, self._train_y, self._test_x, self._test_y = data


	def run(self, k, train_number, test_number, eigenspace_dimension):

		if eigenspace_dimension is not None :
			train_x, test_x = self.eigenprojection(eigenspace_dimension)
			train_x = train_x[:train_number]
			test_x = test_x[:test_number]
		else:
			train_x = self._train_x[:train_number]
			test_x = self._test_x[:test_number]


		accuracy = 0.0

		start_time = time.time()

		for test_img_idx in np.ndindex(test_x.shape[:1]):

			distance = np.linalg.norm(test_x[test_img_idx] - train_x, axis=1)	
			# get euclidean distance 

			sorted_indices = np.argsort(distance)	
			# sorted by order of min dist

			pred_class = stats.mode(self._train_y[sorted_indices[:k]])[0][0]
			# predicted class

			if self._test_y[test_img_idx] == pred_class:
				accuracy += 1
			# count correct answer


		end_time = time.time()

		f = open("./record.txt", "a")

		print("\nTrain number:", train_number, "\t Test number:", test_number)
		f.write("\nTrain number: " + str(train_number)+ "\t Test number:" + str(test_number)+ "\n")

		if eigenspace_dimension == None:
			print("\t[Result] k=", k, "\tTotal Acc is ", accuracy/test_number, "\tTime:", end_time - start_time, "seconds")
			f.write("\t[Result] k="+str(k)+"\tTotal Acc is " +str(accuracy/test_number)+ "\tTime:"+ str(end_time - start_time)+ "seconds\n")

		else:
			print("\t[Result] k=", k, "\teigenspace dimension=", eigenspace_dimension,"\tTotal Acc is ", accuracy/test_number, "\tTime:", end_time - start_time, "seconds")
			f.write("\t[Result] k="+str(k)+"\teigenspace dimension="+str(eigenspace_dimension)+"\tTotal Acc is "+str(accuracy/test_number) + "\tTime:" + str(end_time - start_time)+ "seconds\n")
		f.close()

	def eigenprojection(self, eigenspace_dim = 2):

		cov_matrix = np.cov((self._train_x-np.mean(self._train_x)).T)
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
		eigenvectors =eigenvectors.T[:eigenspace_dim].T
		dot_producted_unit_vector = np.dot(eigenvectors, eigenvectors.T)

		projectioned_train = np.dot(self._train_x, dot_producted_unit_vector)
		projectioned_test = np.dot(self._test_x, dot_producted_unit_vector)
 
		return (projectioned_train, projectioned_test)
 
		# project data on eigenspace		
