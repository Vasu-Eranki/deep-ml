import math
def bce_for_a_single_sample(y:int,p:float,epsilon)->float:
	# y*log(p)+(1-y)*log(1-p)
	return y*math.log(p) + (1-y)*math.log(1-p)
	
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
	loss = [bce_for_a_single_sample(i,j,epsilon) for i,j in zip(y_true,y_pred)]
	return -sum(loss)/len(loss)
	