import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    clipped_probabilities = np.clip(predicted_probs,epsilon, 1-epsilon)
    product_of_labels_and_probabilities = true_labels*clipped_probabilities
    nonzero_entries = product_of_labels_and_probabilities[product_of_labels_and_probabilities>0]
    return -np.mean(np.log(nonzero_entries))