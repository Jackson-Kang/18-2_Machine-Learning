from modules import data_parser
from modules import rbf

if __name__ == "__main__":

	data = data_parser.load_cis_data(["./data/cis_train1.txt", "./data/cis_train2.txt", "./data/cis_test.txt"], index = 1)
	
	rbf_instance = rbf.RBF(data = data)
	rbf_instance.run(k = 2, var = 2)
