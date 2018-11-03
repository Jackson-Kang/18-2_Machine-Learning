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


	def run(self, k, train_number, test_number, eigenspace_dimension, mode):

		self._train_x = self._train_x[:train_number]
		self._train_y = self._train_y[:train_number]
		self._test_x = self._test_x[:test_number]
		self._test_y = self._test_y[:test_number]

		if eigenspace_dimension is not None :
			if mode == "PCA":
				self._train_x, self._test_x = self.PCA(eigenspace_dimension)
			elif mode == "LDA":
				self._train_x, self._test_x = self.LDA(eigenspace_dimension)			


		accuracy = 0.0

		start_time = time.time()

		for test_img_idx in np.ndindex(self._test_x.shape[:1]):

			distance = np.linalg.norm(self._test_x[test_img_idx] - self._train_x, axis=1)	
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
			print("\t[Result] mode=", mode,  "\tk=", k, "\teigenspace dimension=", eigenspace_dimension,"\tTotal Acc is ", accuracy/test_number, "\tTime:", end_time - start_time, "seconds")
			f.write("\t[Result] mode="+ str(mode)+ "\tk="+str(k)+"\teigenspace dimension="+str(eigenspace_dimension)+"\tTotal Acc is "+str(accuracy/test_number) + "\tTime:" + str(end_time - start_time)+ "seconds\n")
		f.close()


	def _eigen_decomposition(self, data, eigenspace_dim, mode):

		if mode == "PCA":
			cov_matrix = np.cov((data-np.mean(data, axis = 0)).T)
			eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
			eigenvectors =eigenvectors.T[:eigenspace_dim].T
			return eigenvectors


		elif mode == "LDA":
			eigenvalues, eigenvectors = np.linalg.eig(data)
			eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
			eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
			

			eigenvectors = np.array([eigen_pairs[i][1] for i in range(eigenspace_dim)])
		
			return eigenvectors.T


	def PCA(self, eigenspace_dim = 2):

		eigenvectors = self._eigen_decomposition(data = self._train_x, eigenspace_dim = eigenspace_dim, mode="PCA")
		projectioned_train = np.dot(self._train_x, eigenvectors)
		projectioned_test = np.dot(self._test_x, eigenvectors)
 
		return (projectioned_train, projectioned_test)
 
		# project data on eigenspace	



	def LDA(self, eigenspace_dim = 2):

		class_number = 10
		class_feature_number = np.array([])
		variance_within_class = np.zeros(784*784).reshape(-1, 784)
		variance_between_class = np.zeros(784*784).reshape(-1, 784)

		global_mean = np.mean(self._train_x, axis = 0)

		for i in np.ndindex(class_number):
			class_features = self._train_x[self._train_y == i]
			class_mean = np.mean(class_features, axis = 0)
			for j in np.ndindex(class_features.shape[0]):
				features_minus_class_mean = (class_features[j] - class_mean).reshape(784, 1)
				variance_within_class += np.dot(features_minus_class_mean, features_minus_class_mean.T)

			dist_between_class_data = (class_mean - global_mean).reshape((784, 1))
			variance_between_class += (class_features.shape[0] * np.dot(dist_between_class_data, dist_between_class_data.T))

		eigenvectors = self._eigen_decomposition( data = np.dot(np.linalg.pinv(variance_within_class), variance_between_class), eigenspace_dim = eigenspace_dim, mode = "LDA")

		projected_train = np.dot(self._train_x, eigenvectors)
		projected_test = np.dot(self._test_x, eigenvectors)




		return projected_train, projected_test





















