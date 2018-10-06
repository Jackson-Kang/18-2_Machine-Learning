from modules import data_loader
from modules import kNN

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

	k = take_argument()

	images = data_loader.load_data('mnist.pkl.gz')
	knn = kNN.knn_init(data=images, k_value=k)

	knn.run()

