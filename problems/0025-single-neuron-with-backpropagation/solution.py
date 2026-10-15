import numpy as np
def sigmoid(x:np.ndarray)->np.ndarray:
	return 1/(1+np.exp(-x))
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	mse_values = []
	for _ in range(epochs):
		weighted_sum = np.sum(features*initial_weights,axis=1) + initial_bias
		output = sigmoid(weighted_sum)
		error = (output-labels)
		mse = np.mean(error**2)
		mse_values.append(round(mse,4))
		derivative_wrt_weight =	np.sum(2*error*output*(1-output)*features.T,axis=1)/len(features)
		derivative_wrt_bias = np.sum(2*error*output*(1-output))/len(features)
		updated_weights = initial_weights - learning_rate*derivative_wrt_weight
		updated_bias = initial_bias - learning_rate*derivative_wrt_bias
		initial_weights = updated_weights
		initial_bias = updated_bias
	return updated_weights.tolist(), updated_bias, mse_values