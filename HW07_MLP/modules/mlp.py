from scipy.stats import truncnorm

import numpy as np


class MLP:

	def __init__(self, data, hidden_node_num, number_of_class):

		'''
			<MNIST Classifier>

				Used Multi-layer perceptron (MLP)
								
				- # of layer: 2

		'''

		self._train_x, self._train_y, self._test_x, self._test_y = data

		self._train_y = self._convert_label_to_one_hot(label = self._train_y)
		self._test_y = self._convert_label_to_one_hot(label = self._test_y)

		input_node_num = self._train_x.shape[1]
		hidden_node_num = hidden_node_num
		output_node_num = number_of_class

		# weight initialization - use "xavier initializer"
		self._W_input_to_hidden = self._xavier_initializer(input_shape = input_node_num, input_node_num = input_node_num, output_node_num = hidden_node_num)
		self._W_hidden_to_output = self._xavier_initializer(input_shape = hidden_node_num, input_node_num = hidden_node_num, output_node_num = output_node_num)


	def run(self, learning_rate, epochs, log_step):

		self._train(epochs = epochs, learning_rate = learning_rate, log_step = log_step)
		self._test()

	
	def _train(self, epochs, learning_rate, log_step):

		loss = 0

		print("\nStart training...")
		for epoch in range(epochs):
			print("\tEpoch %02d" %(epoch+1))
			for i in range(self._train_x.shape[0]):
				output_prob, a1, b1, a2 = self._forward_propagation(input_vector = self._train_x[i])
				loss += self._backpropagation(output_prob = output_prob, label = self._train_y[i], learning_rate = learning_rate, a1 = a1, b1 = b1, a2 = a2) 

				if (i+1) % log_step == 0:
					loss /= log_step
					print("\t\tStep %05d" % (i+1), "\t\tloss = {0:.6f}".format(loss))
					loss = 0
				

	def _test(self):

		correct_count = 0

		print("\n Start testing...")

		for i in range(self._test_x.shape[0]):
			output_prob, _, _, _ = self._forward_propagation(input_vector = self._test_x[i])
			predicted_label = np.argmax(output_prob)
			target_label = np.argmax(self._test_y[i])

			if predicted_label == target_label:
				correct_count += 1

		acc = float(correct_count) / self._test_x.shape[0] 

		print("\t\t[Result] Accuracy: {0:.4f}".format(acc), "Correct count: ", correct_count)
		print("\nFinish!\n")



	def _forward_propagation(self, input_vector):

		'''
			<forward propagation>

				input: 1 x 784 vector (vectorized 1-D MNIST image)
				output: probability of each class
		'''		

		# input to hidden layer 
		a1 = input_vector

		b2 = np.matmul(a1, self._W_input_to_hidden)
		a2 = self._sigmoid_activation(input_vector = b2)
		

		# hidden layer to output layer
		output = np.matmul(a2, self._W_hidden_to_output)
		output = self._softmax(input_vector = output)	

		return output, a1, a2, b2


	def _backpropagation(self, output_prob, label, learning_rate, a1, b1, a2):

		'''
			<backpropagation>

				input: probability of each class
				output: loss from one MNIST image
		'''
		
		loss = self._cross_entropy_loss(input_vector = output_prob, label = label)

		output_delta = output_prob - label
		hidden_delta = self._derivative_of_sigmoid_activation(input_vector = a2) * np.sum(output_delta * self._W_hidden_to_output, axis = 1)

		input_to_hidden_gradient = a1.reshape(-1, 1).dot(hidden_delta.reshape(-1, 1).T)
		hidden_to_output_gradient = output_delta*a2.reshape(-1, 1)

		self._W_hidden_to_output -= learning_rate * hidden_to_output_gradient
		self._W_input_to_hidden -= learning_rate * input_to_hidden_gradient
	
	
		return loss


	def _cross_entropy_loss(self, input_vector, label):
		return -np.sum(np.dot(label, np.log(input_vector).T))

	def _sigmoid_activation(self, input_vector):
		# sigmoid activation function
		return 1 / (1 + np.exp(- input_vector))
	
	def _derivative_of_sigmoid_activation(self, input_vector):
		return self._sigmoid_activation(input_vector = input_vector) * self._sigmoid_activation(input_vector = 1 - input_vector)

	def _softmax(self, input_vector):
		# softmax activation function
		return np.exp(input_vector) /np.sum(np.exp(input_vector))

	def _xavier_initializer(self, input_shape, input_node_num, output_node_num):
		return np.random.rand(input_shape, output_node_num)*np.sqrt(1/(input_node_num + output_node_num))

	def _convert_label_to_one_hot(self, label):

		# convert int label to one hot vector 
		#                                         (one hot encoder)

		one_hot = np.zeros((label.shape[0], 10), dtype = int)

		for i in range(one_hot.shape[0]):
			one_hot[i][label[i]] = 1

		return one_hot

