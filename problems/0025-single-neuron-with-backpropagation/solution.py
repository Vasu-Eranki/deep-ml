import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    mse_values = []
    for _ in range(epochs):
        number_of_examples = len(features)
        product_of_weights = features@initial_weights.T+initial_bias
        sigmoid_activation = 1/(1+np.exp(-product_of_weights))
        mse_loss = np.mean((labels-sigmoid_activation)**2)

        derivate_wrt_w = -2*(labels-sigmoid_activation)*(sigmoid_activation)*(1-sigmoid_activation)*features.T
        derivate_wrt_b = -2*(labels-sigmoid_activation)*(sigmoid_activation)*(1-sigmoid_activation)
        updated_weights = initial_weights  - learning_rate*np.sum(derivate_wrt_w,axis=-1)/number_of_examples
        updated_bias = initial_bias -learning_rate*np.sum(derivate_wrt_b)/number_of_examples
        mse_values.append(mse_loss)
        initial_bias = updated_bias
        initial_weights = updated_weights
    return updated_weights.tolist(), updated_bias, mse_values