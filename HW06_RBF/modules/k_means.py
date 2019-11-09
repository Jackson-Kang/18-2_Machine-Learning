import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import decomposition


class k_Means:

	"""
	class 	"k_Means"

	args:
		data
			: the dataset (x)		# (N, D) Matrix
		k
			: the # of clusters		# scalar
	"""



	def __init__(self, data, k):
		self._train_x = data
		self._k = k


	def _init_centroid(self, data):

		"""
		the randomly-selected initial centroids
		
		args:
			data
				: the dataset (x)					# (N, D) Matrix	
		returns:
			centroids
				: the randomly-selected initial centroid vectors	# (K, D) Matrix
			x_assignee
				: the dataset						# (N, D) Matrix
		"""

		centroids = data[:self._k]
		x_assignee = data

		return centroids, x_assignee


	def _expectation(self, centroids, x_assignee):

		"""
		clusters the datapoints given fixed-centroids from initial centroids or "maximization" process

		args:
			centroids
				: the centroids vector						# (K, D) Matrix
			x_assignee
                                : the dataset                                           	# (N, D) Matrix

		returns:
			r_list
				: the estimated membership for each cluster given centroids	# (N, K) Matrix

		"""


		# Assign data plot to each cluster
		r_list = np.array([], dtype= np.int)

		N, D = x_assignee.shape

		for i in np.ndindex(x_assignee.shape[0]):
			r = np.zeros(self._k, dtype=np.int)
			dist = np.linalg.norm(x_assignee[i] -centroids, axis = 1)
			r[np.argmin(dist)]=1
			r_list = np.append(r_list, r)
		r_list = r_list.reshape((-1, self._k))
	
		return r_list



	def _maximization(self, r_list, x_assignee):
		"""
		update the centroids given updated membership

		args:
			r_list
				: the estimated cluster-membership from "expectation"				# (N, K) Matrix
			x_assignee:
				: the dataset									# (N, D) Matrix

		returns:
			centroids
				: the updated centroid vectors given estimated cluster-membership		# (K, D) Matrix

		"""


		# Calculate mu
		centroids = np.array([])
		r_list = r_list.T
		_, dim = x_assignee.shape

		for i in np.ndindex(r_list.shape[0]):
			summation = np.zeros(x_assignee.shape[1])
			count = 0
			for j in np.ndindex(r_list.shape[1]):
				if r_list[i, j] == 1:
					summation += x_assignee[j]
					count+=1
			mu = summation * (1/count)

			centroids=np.append(centroids, mu)

		return centroids.reshape(-1, dim)



	def train(self):
		"""
		train the k-Means algorithm

		args:
			None

		returns:
			r
				: the converged cluster-membership			# (N, K) Matrix
			centroids
				: the converged centroid vectors			# (N, D) Matrix
		"""

		centroids, x_assignee = self._init_centroid(data=self._train_x)
		count = 0
		
		prev_r = np.array([])

		while True:
			r = self._expectation(centroids = centroids, x_assignee = x_assignee)
			centroids = self._maximization(r_list = r, x_assignee = x_assignee)

			if len(prev_r) is not 0:
				if (float(np.mean(np.equal(r, prev_r))) ==  1.0):
					count +=1
				else: 
					count=0
			if count == 10:
				break
			prev_r = r


		return r, centroids

