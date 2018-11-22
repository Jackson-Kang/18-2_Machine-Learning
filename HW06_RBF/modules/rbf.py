import numpy as np
from modules import k_means

class RBF:

	def __init__(self, data):
		print("start Radial Basis Function...")
		self._train_x, self._train_y= data


	def _basis_kernel(self, x, k, cluster_mean, cluster_variance):

		print(x.shape[0])
		basis = np.zeros((x.shape[0], 1))
		#print(basis)
		for i in range(x.shape[0]):
			basis[i] = np.exp(np.dot(np.dot(x[i], np.linalg.pinv(cluster_variance)), x[i].T))

		return basis


	def _train(self, k, var):

		kmeans_instance = k_means.k_Means(data=(self._train_x, self._train_y))
		cluster_means, cluster_variances = kmeans_instance.run(k = k)

		value = 0
		basis_matrix = []

		weighted_sum = np.zeros((self._train_x.shape[0], k))

		for i in range(k):
			basis = self._basis_kernel(x = self._train_x, k = k, cluster_mean = cluster_means[i], cluster_variance = cluster_variances[i])
			w = np.dot(np.linalg.pinv(basis), self._train_y)
			basis_matrix.append(w)


		basis_matrix = np.asarray(basis_matrix)
		print(basis_matrix.shape)



	def _test(self):
		pass		



	def run(self, k, var):

		self._train(k = k, var = var)
