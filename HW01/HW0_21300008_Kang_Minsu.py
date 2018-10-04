'''
	21300008 Kang MinSu
	
	HW #0
'''

import numpy as np



def problem1():

	# problem 1:  M = A * B (matmul)

	A = np.random.rand(3, 4)
	B = np.random.rand(4, 3)
	answer_for_problem_1 = np.matmul(A, B)

	print("[Problem 1]") 

	print("\tA =", A)
	print("\tB = ", B)

	print("\tM = A * B =", answer_for_problem_1)
	print()

def problem2():

	# problem 2: M = A X B (element-wise multiplication)

	A = np.random.rand(3, 4)
	B = np.random.rand(3, 4)
	answer_for_problem2 = np.multiply(A, B)

	print("[Problem 2]")
	print("\tA =", A)
	print("\tB = ", B)
	print("\tM = A X B=", answer_for_problem2)
	print()

def problem3():

	# problem 3: calculate y = Ax + b

	A = np.array(np.ones((3, 4)))
	x = np.array(np.ones((4, 1)))
	b = np.array(np.ones((3, 1)))

	y = np.tanh(np.dot(A, x) + b)

	print("[Problem 3]")
	print("\ty = ", y)



if __name__ == "__main__":
	problem1()
	problem2()
	problem3()
