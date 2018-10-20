from modules import data_loader
from modules import kNN

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

import argparse


# Machine Learning HW#3 

#  - implement k-Nearest Neighbor(kNN)  and use Random Forest
#  - implemented by Jackson Kang, 2018. 10. 05.

# take argument from command-line

def take_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument("--k", help="input initial value of \'k\'", required=True)
	args = parser.parse_args()
	k = int(args.k)
	return k

if __name__ == "__main__":

	
	train_data_number_list = [1000, 2000, 5000, 10000, 25000, 50000]
	test_data_number_list = [100, 200, 500, 1000, 2000, 5000, 10000]

	k_list = [1, 5, 10]
	eigenspace_dimension_list = [2, 5, 10]


	images = data_loader.load_data('mnist.pkl.gz')
	knn = kNN.knn_init(data=images)
	

	for train_number in train_data_number_list:
		for test_number in test_data_number_list:
			for k in k_list:
				knn.run(k=k, train_number= train_number, test_number = test_number, eigenspace_dimension = None)
	# raw image


	for train_number in train_data_number_list:
		for test_number in test_data_number_list:
			for k in k_list:
				for eigenspace_dimension in eigenspace_dimension_list:
					knn.run(k=k, train_number = train_number, test_number = test_number, eigenspace_dimension = eigenspace_dimension)
	# eigenspace projection


	clf = RandomForestClassifier(n_estimators = 200, max_depth = 500)
	clf.fit(images[0], images[1])

	#print(clf.feature_importances_)

	acc = np.mean(np.equal(clf.predict(images[2]), images[3]))
	print("Accuracy is ", acc)

	


	print("finish!")

