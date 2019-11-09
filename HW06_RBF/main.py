from modules import utils
from modules import RBFN

import numpy as np
import time as t

def do_experiments(m, s, mode="all"):

	mse_performance, mse_time, acc_performance, acc_time = 0, 0, 0, 0

	if mode == "all" or mode== "regression": 
		print("m=", m, "\t\ts=", s)
		fa_rbf = RBFN.Radial_Basis_Function_Network(train=fa_train, test=fa_test, M=m, s=s, metric = "MSE")

		mse_start = t.time()
		fa_predicted_y, mse_performance = fa_rbf.run()
		mse_time = t.time() - mse_start

	if mode == "all" or mode == "classification":
		print()
		cis_rbf = RBFN.Radial_Basis_Function_Network(train=cis_train, test=cis_test, M=m, s=s, metric="Accuracy")

		class_start = t.time()
		cis_predicted_y, acc_performance= cis_rbf.run()
		class_time = t.time()-class_start

	return mse_performance, mse_time, acc_performance, acc_time




if __name__ == "__main__":


	for file_index in [0, 1]:


		_, cis_train_y = cis_train = utils.load_data(["./data/cis_train1.txt", "./data/cis_train2.txt", "./data/cis_test.txt"], index = file_index)
		cis_test_x, cis_test_y = cis_test = utils.load_data(["./data/cis_train1.txt", "./data/cis_train2.txt", "./data/cis_test.txt"], index = 2)

		_, fa_train_y = fa_train = utils.load_data(["./data/fa_train1.txt", "./data/fa_train2.txt", "./data/fa_test.txt"], index=file_index)
		_, fa_test_y = fa_test = utils.load_data(["./data/fa_train1.txt", "./data/fa_train2.txt", "./data/fa_test.txt"], index=2)


		mse_best = 100
		acc_best = 0

		m_list = range(1, 21)
		s_list = np.arange(0.1, 5.1, 0.1)


		mse_list = []
		mse_time_list = []

		acc_list = []
		acc_time_list = []	

		for m in m_list:
			for s in s_list:

				mse_performance, mse_time, acc_performance, acc_time=do_experiments(m=m, s=s)

				if mse_best > mse_performance:
					mse_best_m = m
					mse_best_s = s
					mse_best = mse_performance
				if acc_best < acc_performance:
					acc_best_m = m
					acc_best_s = s
					acc_best = acc_performance

		m_variants = []
		for m in m_list:
			_, _, acc_performance, acc_time = do_experiments(m=m, s=acc_best_s, mode="classification")
			mse_performance, mse_time, _, _ = do_experiments(m=m, s=mse_best_s, mode="regression")
			m_variants.append([m, acc_performance, acc_time, mse_performance, mse_time])

		s_variants = []
		for s in s_list:
			_, _, acc_performance, acc_time = do_experiments(m=acc_best_m, s=s, mode="classification")
			mse_performance, mse_time, _, _  = do_experiments(m=mse_best_m, s=s, mode="regression")
			s_variants.append([s, acc_performance, acc_time, mse_performance, mse_time])

		m_variants = np.asarray(m_variants)
		s_variants = np.asarray(s_variants)
		
		for i in range(1, 5):
			if i == 1 or i == 2:
				train_sample_number = cis_train_y.shape[0] 

			else:
				train_sample_number = fa_train_y.shape[0]

			utils.draw_variants_graph(x_axis=m_variants[:, 0], y_axis=m_variants[:, i], train_sample_number=train_sample_number, m_or_s="M", idx=i)


		for i in range(1, 5):
			if i == 1 or i == 2:
				train_sample_number = cis_train_y.shape[0]
			else:
				train_sample_number = fa_train_y.shape[0]

			utils.draw_variants_graph(x_axis=s_variants[:, 0], y_axis=s_variants[:, i], train_sample_number=train_sample_number, m_or_s="S", idx=i)


		print()
		print("Try with best model for regression...")
		fa_rbf = RBFN.Radial_Basis_Function_Network(train=fa_train, test=fa_test, M=mse_best_m, s=acc_best_s, metric = "MSE")
		fa_predicted_y, mse_performance = fa_rbf.run()
		print()

		print("Try with best model for classification...")
		cis_rbf = RBFN.Radial_Basis_Function_Network(train=cis_train, test=cis_test, M=acc_best_m, s = acc_best_s, metric="Accuracy")
		cis_predicted_y, acc_performance= cis_rbf.run()
		print()
		print()


		print("The best m, s for regression is", mse_best_m, mse_best_s)
		utils.draw_best_model_graph(path_to_save='../../lab_desktop/fa_best'+str(fa_train_y.shape[0])+'.png', x_axis=np.arange(0.0, 1.0, 1/fa_test_y.shape[0]), predicted_y=fa_predicted_y, label_y=fa_test_y, task="regression", train_sample_number=fa_train_y.shape[0])

		print()

		print("The best m, s for classification is", acc_best_m, acc_best_s)
		utils.draw_best_model_graph(path_to_save='../../lab_desktop/cis_best'+str(cis_train_y.shape[0])+'.png', x_axis=cis_test_x, predicted_y=cis_predicted_y, label_y=cis_test_y, task="classification", train_sample_number=cis_train_y.shape[0])
