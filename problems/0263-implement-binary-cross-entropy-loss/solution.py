import math
def calculate_log_of_value(y:int,p:float):
	# y*log(p) + (1-y)*log(1-p)
	result = y*math.log(p) + (1-y)*math.log(1-p)
	return result 


def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	results = [calculate_log_of_value(y,p) for y,p in zip(y_true,y_pred)]
	sum_of_results = sum(results)
	mean_of_results = sum_of_results/len(y_true)
	return -1*mean_of_results

	