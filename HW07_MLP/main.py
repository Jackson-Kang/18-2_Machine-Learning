from modules import data_loader
from modules import mlp

import numpy as np



# Machine Learning HW#7

#  - Implement MLP and apply it to MNIST
#  - implemented by Jackson Kang, 2018. 11. 20.



if __name__ == "__main__":

	
	images = data_loader.load_data('mnist.pkl.gz')

	mlp_instance = mlp.MLP(data = images, hidden_node_num = 100, number_of_class = 10)
	mlp_instance.run(learning_rate = 0.00001, epochs = 1000, log_step = 1000)
