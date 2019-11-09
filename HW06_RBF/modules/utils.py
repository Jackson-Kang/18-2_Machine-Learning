import matplotlib.pyplot as plt
import numpy as np

def load_data(path_list, index):

	"""
	load dataset from each file

	args:
		path_list	[list of string]
			: the "N-length" path list	(ex: ["file1.txt", "file2.txt", ..., "fileN.txt"]
		index		[int]
			: which file you want to select from path_list
	returns:
		x
			: the input vectors, x
		y
			: the label vectors, y
	"""

	x, y = _parse_data(path_list[index])

	return x, y

def draw_variants_graph(x_axis, y_axis, train_sample_number, m_or_s, idx):

	if idx == 1:
		mode = "Accuracy"
	elif idx ==2:
		mode = "time_ACC"
	elif idx == 3:
		mode = "MSE"
	elif idx == 4:
		mode = "time_MSE"

	if m_or_s == "m":
		x_axis = x_axis.astype(int)

	color = [' ', 'C0', 'C1', 'C2', 'C3']

	fig, ax = plt.subplots()
	title= "Distribution of "+mode+"\'s varation with the increase of " + m_or_s + " when " + str(train_sample_number) + " samples"

	ax.plot(x_axis, y_axis, color[idx], label=mode)
	plt.xlabel(m_or_s)
	plt.ylabel(mode)
	ax.set_title(title)
	plt.savefig('/home/minsu/minsu/lab_desktop/'+m_or_s+"_"+mode+"_"+str(train_sample_number)+".png")




def draw_best_model_graph(path_to_save, x_axis, predicted_y, label_y, task, train_sample_number):
	"""
	save the results for analysis

	args:
		path_to_save 	[STRING]
			: target filepath for saving graph					
		x_axis		[SEQUENCE OF FLOAT]
			: the x-axis		(the sequence of # of basis function, or s)			# a sequence	[T, 1]
		y_axis		[SEQUENCE OF FLOAT]
			: the y-axis		(the predicted results) 					# a sequence	[T, 1]
		task		[STRING]
			: regression		(the graph of approximated function)	
			: classification	(the graph of approximated circle)
	returns:
		None	
			Please go to "path_to_save" and see the results 
	"""

	fig, ax = plt.subplots()
	title = str(train_sample_number) + " samples"

	if task == "regression":

		title = "Function Approximation Results: " + title

		ax.plot(x_axis, predicted_y, 'C3', label="predicted")
		ax.plot(x_axis, label_y, 'C0', label="label")
		ax.legend(loc="upper left")


	elif task == "classification":
		title = "Circle in the square Results: " + title		

		indices = predicted_y.repeat(2).reshape(-1, 2)


		x1 = x_axis[indices==1].reshape(-1, 2)
		x_1= x_axis[indices==-1].reshape(-1, 2)

		ax.scatter(x1[:, 0], x1[:, 1], c='w')
		ax.scatter(x_1[:, 0], x_1[:, 1], c='black')

	
	ax.set_title(title)
	plt.savefig(path_to_save)


def _parse_data(path):

	"""
	parse data
	
	args:
		path
			: selected file that contains data

	returns:
		x
			: the input vectors, x
		y
			: the output vectors, y
	"""

	with open(path, "r", encoding="utf-8") as f:
		x_list, labels = [], []

		for line in f.readlines():
			data = line.split("\t")
			x_list.append([float(temp) for temp in data[:-1]])
			labels.append(float(data[-1][:-1]))
	
	return np.asarray(x_list), np.asarray(labels).reshape(-1, 1)
