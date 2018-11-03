from modules import data_loader
from modules import kNN

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

import argparse


# Machine Learning HW#3 

#  - implement k-Nearest Neighbor(kNN)  and use Random Forest
#  - implemented by Jackson Kang, 2018. 10. 05.



if __name__ == "__main__":	

	images = data_loader.load_data('mnist.pkl.gz')


	k_list = [3]
	eigenspace_list = [2, 3, 5, 9]
	train_number_list = [1000, 2000, 5000, 10000, 25000, 50000]
	test_number_list = [100, 500, 1000, 5000, 10000]


	for k in k_list:
		for dim in eigenspace_list:
			for train_number in train_number_list:
				for test_number in test_number_list:

					if train_number < test_number:
						continue

					for mode in ["PCA", "LDA"]:
						knn = kNN.knn_init(data = images)
						knn.run(k=k, train_number = train_number, test_number = test_number, eigenspace_dimension=dim, mode = mode)


