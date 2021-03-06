'''

	<Machine Learning Homework #5>


	- Implement and apply LDA to MNIST to find 2, 3, 5, 9 dim spaces.

	- Compare them to the corresponding eigenspaces in classification accuracy (with kNN and random forest)
	
	
	- Written by 21300008 Kang Min-Su




	<Step to execute this code>

		1. install packages by following command
	
			pip install -r requirements.txt

		2. python main.py

			(recommend that you use python 3.6 version)

		3. check log or record.txt so that you can see the performances

	
	* if you want to check whether this code works or not, 
	  	just simply change the train and number in 'train_number_list' and 'test_number_list' into smaller value *

	<Cautions>

        This source code was implemented in Ubuntu server. If there's an error, please call me.

'''
import os
from sys import path
from os import getcwd
path.append(getcwd())

from modules import data_loader
from modules import modules

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

import argparse


if __name__ == "__main__":	


	images = data_loader.load_data('mnist.pkl.gz')

	k_list = [1, 3, 5, 10]
	eigenspace_list = [2, 3, 5, 9]
	train_number_list = [50000]
	test_number_list = [10000]
	mode_list = ["PCA", "LDA"]

	for k in k_list:
		for dim in eigenspace_list:
			for train_number in train_number_list:
				for test_number in test_number_list:

					if train_number < test_number:
						continue

					for mode in mode_list:
						module = modules.module_init(data = images)
						module.knn_run(k = k, train_number = train_number, test_number = test_number, eigenspace_dimension=dim, mode = mode)
	 
	for dim in eigenspace_list:
		for train_number in train_number_list:
			for test_number in test_number_list:
				if train_number < test_number:
					continue
				
				for mode in mode_list:
					module = modules.module_init(data =images)
					module.random_forest_run(train_number = train_number, test_number = test_number, eigenspace_dimension = dim, mode = mode)

