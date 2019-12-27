from modules import modules, data_loader

import matplotlib.pyplot as plt


if __name__ == "__main__":
	images = data_loader.load_data("mnist.pkl.gz")
	module = modules.module_init(data=images)
	projected_pca_train_data, _ = module.PCA(eigenspace_dim=2)
	projected_lda_train_data, _ = module.LDA(eigenspace_dim=2)

	_, train_y, _, _= images
	
	for mode in ["PCA", "LDA"]:

		fig, ax = plt.subplots()

		for j in range(0, 10):
			if mode == "PCA":
				class_data = projected_pca_train_data[train_y==j] 
			else:
				class_data = projected_lda_train_data[train_y==j]

			x1_axis = class_data[:, 0]
			x2_axis = class_data[:, 1]

			ax.plot(x1_axis, x2_axis, marker='.', linestyle='', label=str(j))
			
		plt.title("Scatter-plot of MNIST projected on 2-dim eigenspace using "  + mode)
		ax.legend(loc='upper left')
		plt.savefig("./"+mode+".png")
