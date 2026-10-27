import numpy as np
def softmax(logits:list[float])->list[float]:
    maximum_value = np.max(logits,axis=-1,keepdims=True)
    scaled_scores = np.exp(logits-maximum_value)
    softmax_scores = scaled_scores/np.sum(scaled_scores,axis=-1,keepdims=True)
    return softmax_scores

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    indices = [i for i in range(len(logits))]
    softmax_scores = softmax(logits)
    probability_scores_sorted_in_descending_order = np.argsort(-softmax_scores,axis=-1) # sort in descending
    sorted_p_scores = softmax_scores[probability_scores_sorted_in_descending_order]
    cumsum = np.cumsum(sorted_p_scores, axis=-1) >=p
    first_idx_which_satisfies_nucleus = np.argmax(cumsum, axis=-1)
    indices_in_nucleus = probability_scores_sorted_in_descending_order[:first_idx_which_satisfies_nucleus+1].tolist()
    indices_not_in_nucleus = [i for i in indices if i not in indices_in_nucleus]
    softmax_scores[indices_not_in_nucleus] = 0
    softmax_scores = softmax_scores/np.sum(softmax_scores,axis=-1,keepdims=True)
    return softmax_scores