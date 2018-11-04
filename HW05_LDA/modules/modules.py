from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import time

class module_init:

	'''
		kNN and Random forest implementation

			written by 21300008 Kang Min-Su
	'''


	def __init__(self, data):
		self._train_x, self._train_y, self._test_x, self._test_y = data
		# data init

	def knn_run(self, k, train_number, test_number, eigenspace_dimension, mode):

		'''
			do kNN
		'''

		self._data_preprocessor(train_number = train_number, test_number = test_number, eigenspace_dimension = eigenspace_dimension, mode = mode)

		accuracy = 0.0
		start_time = time.time()

		for test_img_idx in np.ndindex(self._test_x.shape[:1]):

			distance = np.linalg.norm(self._test_x[test_img_idx] - self._train_x, axis=1)	
			# calculate distances from one test data point to train data samples

			sorted_indices = np.argsort(distance)	
			# sort index of distances in increasing order

			pred_class = stats.mode(self._train_y[sorted_indices[:k]])[0][0]
			# get candidate class labels and select the mode value of them

			if self._test_y[test_img_idx] == pred_class:
				accuracy += 1
			# calculate acc

		time_spent = time.time() - start_time
		accuracy /= test_number

		self._result_writer(eigenspace_dimension=eigenspace_dimension, k=k, accuracy=accuracy, time_spent = time_spent, train_number = train_number, test_number = test_number, mode = mode) 
		# kNN main



	def random_forest_run(self, train_number, test_number, eigenspace_dimension, mode):

		'''
			do Random Forest
		'''

		self._data_preprocessor(train_number = train_number, test_number = test_number, eigenspace_dimension = eigenspace_dimension, mode = mode)
		
		start_time = time.time()	

		clf = RandomForestClassifier()
		clf.fit(self._train_x.real, self._train_y)

		acc = np.mean(np.equal(clf.predict(self._test_x.real), self._test_y))
		time_spent = time.time() - start_time

		self._result_writer(eigenspace_dimension=eigenspace_dimension, k="Random Forest", accuracy=acc, time_spent = time_spent, train_number = train_number, test_number = test_number, mode = mode)
		


	def _data_preprocessor(self, train_number, test_number, eigenspace_dimension, mode):
		'''
			do PCA or LDA, and split data
		'''
		self._train_x = self._train_x[:train_number]
		self._train_y = self._train_y[:train_number]
		self._test_x = self._test_x[:test_number]
		self._test_y = self._test_y[:test_number]
		# split data 

		if eigenspace_dimension is not None :
			if mode == "PCA":
				self._train_x, self._test_x = self.PCA(eigenspace_dimension)
			elif mode == "LDA":
				self._train_x, self._test_x = self.LDA(eigenspace_dimension)
		# do PCA or LDA according to eigenspace dimension


	def _result_writer(self, eigenspace_dimension, accuracy, train_number, test_number, time_spent, mode, k = None):
		'''
			record learning result 
		'''

		f = open("./record.txt", "a")

		print("\nTrain number:", train_number, "\t Test number:", test_number)
		f.write("\nTrain number: " + str(train_number)+ "\t Test number:" + str(test_number)+ "\n")

		if eigenspace_dimension == None:
			print("\t[Result] k=", k, "\tTotal Acc is ", accuracy, "\tTime:", time_spent, "seconds")
			f.write("\t[Result] k="+str(k)+"\tTotal Acc is " +str(accuracy)+ "\tTime:"+ str(time_spent)+ "seconds\n") 
		else:
			print("\t[Result] mode=", mode,  "\tk=", k, "\teigenspace dimension=", eigenspace_dimension,"\tTotal Acc is ", accuracy, "\tTime:", time_spent, "seconds")
			f.write("\t[Result] mode="+ str(mode)+ "\tk="+str(k)+"\teigenspace dimension="+str(eigenspace_dimension)+"\tTotal Acc is "+str(accuracy) + "\tTime:" + str(time_spent)+ "seconds\n")

		f.close()
		# for recoding model performances


	def _check_variance_percentage(self, eigenvalues, eigenspace_dim, mode):
		
		percentages = eigenvalues / np.sum(eigenvalues)

		total_percentages = np.sum(eigenvalues[:eigenspace_dim]) / np.sum(eigenvalues)

		print("mode: ", mode, "\tvariance percentages:", list(percentages[:eigenspace_dim]), "\t total percentage:", total_percentages)


	def PCA(self, eigenspace_dim):

		'''
			do PCA analysis
		'''
		cov_matrix = np.cov((self._train_x - np.mean(self._train_x, axis = 0)).T)
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
		eigenvectors =eigenvectors.T[:eigenspace_dim].T
		# do eigen-decomposition

		# self._check_variance_percentage(eigenvalues=np.abs(eigenvalues), eigenspace_dim=eigenspace_dim, mode = "PCA")
			# if you want to check variance percentage, decomment this code


		projectioned_train = np.dot(self._train_x, eigenvectors)
		projectioned_test = np.dot(self._test_x, eigenvectors)
		# project data onto n-eigenspace
 
		return (projectioned_train, projectioned_test) 
		# project data on eigenspace	


	def LDA(self, eigenspace_dim):
		''' 
			do LDA analysis
		'''

		class_number = 10
		class_feature_number = np.array([])
		variance_within_class = np.zeros(784*784).reshape(-1, 784)
		variance_between_class = np.zeros(784*784).reshape(-1, 784)

		global_mean = np.mean(self._train_x, axis = 0)

		for i in np.ndindex(class_number):
			class_features = self._train_x[self._train_y == i]
			class_mean = np.mean(class_features, axis = 0)
			# calculate mean of each class

			for j in np.ndindex(class_features.shape[0]):
				features_minus_class_mean = (class_features[j] - class_mean).reshape(784, 1)
				variance_within_class += np.dot(features_minus_class_mean, features_minus_class_mean.T)
				# calculate variance within class to minimize variance between samples in class

			dist_between_class_data = (class_mean - global_mean).reshape((784, 1))
			variance_between_class += (class_features.shape[0] * np.dot(dist_between_class_data, dist_between_class_data.T))
			# calculate variance between classes to maximize variance between classes in sample space

		eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(variance_within_class), variance_between_class))
		# eigen-decompose (SW)^(-1) X (SB)
		eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
		eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

		eigenvalues = np.array([eigen_pairs[i][0] for i in range(len(eigenvalues))])
		eigenvectors = np.array([eigen_pairs[i][1] for i in range(eigenspace_dim)]).T
		# sort (eigenvalue, eigenvector) pair
		
		# self._check_variance_percentage(eigenvalues=eigenvalues, eigenspace_dim=eigenspace_dim, mode = "LDA")
			# if you want to check variance percentage, decomment this code

		projected_train = np.dot(self._train_x, eigenvectors)
		projected_test = np.dot(self._test_x, eigenvectors)
		# project train and test samples

		return projected_train, projected_test


