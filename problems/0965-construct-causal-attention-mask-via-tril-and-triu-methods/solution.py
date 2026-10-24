import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    if "tril":
        ones_matrix = np.ones_like(attn_weights)
        lower_triangular_matrix = np.tril(ones_matrix,k=0)
        attention_weights = attn_weights*lower_triangular_matrix
        attention_weights = attention_weights/np.sum(attention_weights,axis=-1,keepdims=True)
        return attention_weights
    else:
        ones_matrix = np.ones_like(attn_weights)
        upper_triangular_matrix = np.triu(ones_matrix,k=0)
        scaled_scores = np.log(attn_weights)
        scaled_scores = np.where(upper_triangular_matrix==1,scaled_scores,-np.inf)
        exponentiated_scores = np.exp(scaled_scores)
        softmax_scores = exponentiated_scores/np.sum(exponentiated_scores,axis=-1,keepdims=True)
        return softmax_scores
