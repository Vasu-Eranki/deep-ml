import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	x = np.asarray(X)
	inverse_of_matrix = np.linalg.inv(np.matmul(x.T, x))
	result = np.matmul(inverse_of_matrix,x.T)
	result = np.matmul(result, y)
	return result.tolist()