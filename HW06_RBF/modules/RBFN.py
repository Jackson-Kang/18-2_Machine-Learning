import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from modules import k_means


class Radial_Basis_Function_Network:

	def __init__(self, train, test, M, s, kernel="Gaussian", clustering_algorithm="k-Means", metric = "MSE"):

		"""
		[Class]  
			-> Radial_Basis_Funtion_Network
	
		[Mendatory Args]
			train
				: the train dataset with (N=#_of_samples, D=dimension_of_one_sample)			# Matrix
			test 
				: the test dataset with (N, D)								# Matrix
			M
				: the # of basis function 								# Scalar
			s
				: the scale factor of Gaussian Kernel							# Scalar
	
		[Optional Args]
			kernel
				: which kernel will be use to map x to nonlinear space
					default -> Gaussian   (cuz we use Radial Basis Function)
			clustering_algorithm
				: which clustering algorithm will be use to get p(x_j) for cluster j
					default -> k-Means
			metric
				: which task we will solve
					default -> "MSE"
						MSE: 		regression
						Accuracy: 	binary-classification
		"""

		self.__train_x, self.__train_y  = train
		self.__test_x, self.__test_y = test
		self.__M = M
		self.__s = s
		self.__kernel = kernel
		self.__metric = metric
		self.__clustering = self.__apply_clustering(algorithm=clustering_algorithm)

		if self.__metric == "Accuracy":
			self.__train_y[self.__train_y == 0] = -1
			self.__test_y[self.__test_y == 0 ] = -1

	
	def __apply_clustering(self, algorithm="k-Means"):
		"""
		Apply clustering to get p(x_j) for cluster j

		args:
			algorithm
				: which clustering algorithm we will use

		returns:
			if k-Means: 
				one-hot encoded vector which represents the membership for each datapoint		# (N, K) Matrix, where K is # of the cluster
				centroids(mean) in each clusters							# (K, D) Matrix
			else:
				None
		"""

		if algorithm=="k-Means":
			return k_means.k_Means(data=self.__train_x, k=self.__M)

		return None


	def __apply_Gaussian_kernel(self, x, means, s):
		"""
		Map each datapoints to nonlinear Gaussian space

		args:
			x
				: train or test data							# (N, D) Matrix
			means
				: the mean vectors from each clusters					# (M, D) Matrix, where M is # of basis function (i.e., K is same with M)
			s
				: the scail factor which determine width of each basis function
													# scalar
		returns:
			fais
				: the result from applying Gaussian kernel to each datapoints		# (M, N) Matrix		
	
		"""

		N, dim = x.shape
		fais = np.zeros((self.__M, N), dtype=np.float)
		
		for i in np.ndindex(self.__M):
			for j in np.ndindex(N):
				fais[i][j] = np.exp(-s*np.linalg.norm(x[j].reshape(1, dim) - means[i].reshape(1, dim))**2)
		return fais


	def __get_weight(self, fai, t):	
		"""
		Get weight vectors from fai

		args:
			fai
				: the nonlinearly mapped datapoints in the train dataset			# (M, N) Matrix
			t
				: the labels in the train dataset						# (N, 1) vector
		returns:
			weight
				: the trained weight vectors from applying moore-pseudo inverse 		# (M, 1)

		"""


		N = t.shape[0]
		weight = np.dot(np.linalg.pinv(fai).T, t)

		return weight


	def __train(self):
		"""
		Training Radial Basis Fucntion Network from given train dataset

		args:
			None

		returns:
			None
		"""

		_, self.__means  = self.__clustering.train()

		fai_matrix = self.__apply_Gaussian_kernel(x=self.__train_x, means=self.__means, s=self.__s)
		self.__trained_weight = self.__get_weight(fai=fai_matrix, t=self.__train_y)


	def __test(self):
		"""
		Test previously trained Radial Basis Function Network from given test dataset

		args:
			None
		returns:
			y_hat:
				the prediction results from the trained Raidial Basis Function Network
		"""

		fai_matrix = self.__apply_Gaussian_kernel(self.__test_x, means=self.__means, s=self.__s)
		
		energy = np.dot(fai_matrix.T, self.__trained_weight)

		if self.__metric == "MSE":
			y_hat = energy
			performance = ((self.__test_y - y_hat)**2).mean()

		elif self.__metric == "Accuracy":

			y_hat = 1/(1+np.exp(-energy))	
			y_hat[y_hat >= 0.5] = 1
			y_hat[y_hat < 0.5] = -1
			performance = (y_hat == self.__test_y).mean()

		return y_hat, performance


	def run(self):
		"""
		Run the Radial Basis Function Network
		
		args:
			None
		returns:
			prediction restuls of test data
		"""

		self.__train()
		return self.__test()
