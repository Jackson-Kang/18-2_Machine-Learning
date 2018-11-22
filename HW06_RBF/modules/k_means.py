import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
from sklearn import decomposition


class k_Means:

	def __init__(self, data):
		print("start k-Means clustering...")
		self._train_x, self._train_y = data


	def _init_centroid(self, data, label):
		centroids = data[:self._k]
		x_assignee = data
		y_assignee = label

		return centroids, x_assignee, y_assignee

	def _expectation(self, centroids, x_assignee, y_assignee):
		# Assign data plot to each cluster
		r_list = np.array([], dtype= np.int)
		for i in np.ndindex(x_assignee.shape[0]):
			r = np.zeros(self._k, dtype=np.int)
			dist = np.linalg.norm(x_assignee[i]-centroids, axis = 1)
			r[np.argmin(dist)]=1
			r_list = np.append(r_list, r)
		r_list = r_list.reshape((-1, self._k))
	
		return r_list

	def _maximization(self, r_list, x_assignee):
		# Calculate mu
		centroids = np.array([])
		r_list = r_list.T
		for i in np.ndindex(r_list.shape[0]):
			summation = np.zeros(x_assignee.shape[1])
			count = 0
			for j in np.ndindex(r_list.shape[1]):
				if r_list[i, j] == 1:
					summation += x_assignee[j]
					count+=1
			mu = summation * (1/count)
			centroids=np.append(centroids, mu)

		return centroids.reshape(-1, 2)


	def _training(self, data):
		centroids, x_assignee, y_assignee = self._init_centroid(data, self._train_y)
		count = 0
		
		prev_r = np.array([])
		while True:
			r = self._expectation(centroids = centroids, x_assignee = x_assignee, y_assignee = y_assignee)
			centroids = self._maximization(r_list = r, x_assignee = x_assignee)
			if len(prev_r) is not 0:
				if (float(np.mean(np.equal(r, prev_r))) ==  1.0):
					count +=1
				else: 
					count=0
			if count == 10:
				break
			prev_r = r

		return prev_r, centroids


	def run(self, k):
	
		cluster_means = []
		cluster_variances = []

		self._k = k

		r, centroids = self._training(data=self._train_x)	
		which_belongs_to = np.argmax(r, axis = 1)

		for i in range(self._k):
			data_of_cluster_i = self._train_x[np.where(which_belongs_to == i)]
			print(data_of_cluster_i.shape)
			cluster_means.append(np.mean(data_of_cluster_i, axis = 0))
			cluster_variances.append(np.cov(data_of_cluster_i.T))

		return cluster_means, cluster_variances














