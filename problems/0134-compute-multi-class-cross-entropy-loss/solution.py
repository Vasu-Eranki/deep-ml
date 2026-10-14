import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    clipped_probs = np.clip(predicted_probs,epsilon,1)
    product_of_probs_and_labels = true_labels*clipped_probs
    real_nonzero_values = product_of_probs_and_labels[product_of_probs_and_labels!=0]
    cross_entropy_loss = np.log(real_nonzero_values)
    mean_loss = np.mean(cross_entropy_loss)
    return -1*mean_loss