import numpy as np
import random
import imageio
import matplotlib.pyplot as plt


class k_Means:

	def __init__(self, data):
		print("start k-Means clustering...")
		self._train_x, self._train_y, _, _ = data

		#self._train_x = self._train_x[:100]
		#self._train_y = self._train_y[:100]


	def _filter_test_data(self, filtered_list=None):
		if filtered_list is not None:
			indices = np.isin(self._train_y, filtered_list)
			self._train_x = self._train_x[np.where(indices)]
			self._train_y = self._train_y[np.where(indices)]

	def _init_centroid(self):
		centroids = self._train_x[:self._k]
		#x_assignee = self._train_x[self._k:]
		x_assignee = self._train_x
		y_assignee = self._train_y
		#y_assignee = self._train_y[self._k:]

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
		return centroids.reshape(-1, 784)
	

	def _training(self):
		centroids, x_assignee, y_assignee = self._init_centroid()
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


	def _testing(self, r, filtered_list):
		filtered_list = np.array(filtered_list)
		predicted = np.argmax(r, axis = 1)
		predicted[predicted==1] = 3
		predicted[predicted==0] = 9
		acc = np.mean(np.equal(predicted, self._train_y))

		print("\t\t[Results] The maximum ratio is ", acc)

	def _eigenprojection(self, eigenspace_dim = 2):
		cov_matrix = np.cov((self._train_x-np.mean(self._train_x)).T)
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
		
		eigenvectors = eigenvectors.real.T[:eigenspace_dim]
		eigenvectors = self._normalization(data=eigenvectors, norm_range=[-1, 1])
		dot_producted_eigenvectors = np.dot(eigenvectors.T, eigenvectors)
		projectioned_train = np.dot(self._train_x, dot_producted_eigenvectors)

		self._train_x = projectioned_train


	def _normalization(self, data, norm_range):
		print(data.shape)
		for i in np.ndindex(data.shape[0]):
			data[i] = np.interp(data[i], (data[i].min(), data[i].max()), norm_range)
		return data


	def run(self, k, filtered_list =[3, 9], eigenspace_dim=None):

		print("\n\t[Step k=" + str(k) + ", eigenspace_dim="+str(eigenspace_dim)+"]" )

		self._k = k
		self._filter_test_data(filtered_list)
		if eigenspace_dim is not None:
			self._eigenprojection(eigenspace_dim)
		r, centroids = self._training()	
		if len(filtered_list)==k:
			self._testing(r = r, filtered_list = filtered_list)
		if eigenspace_dim is not None:

			centroids = self._normalization(data=centroids, norm_range=[0, 1])

			for i in range(k):
				imageio.imwrite('./results/k_' + str(k) + '_es_' + str(eigenspace_dim) + '_centroid_' + str(i) + '.jpg', centroids[i].reshape((28, 28)))	
		else:
			for i in range(k):
				imageio.imwrite('./results/k_' + str(k) + '_es_' + str(eigenspace_dim) + '_centroid_' + str(i) + '.jpg', centroids[i].reshape((28, 28)))

		print("\t Finish!")

