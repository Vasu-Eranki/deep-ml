import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	seq_len_q,d_k = Q.shape
	raw_scaled_scores = np.matmul(Q, K.T)/(d_k)**0.5
	if mask is not None:
		raw_scaled_scores = np.where(mask==1,raw_scaled_scores,-np.inf)
	maximum_per_row =np.max(raw_scaled_scores, axis=1,keepdims=True)
	exponentiated_scores = np.exp(raw_scaled_scores-maximum_per_row)
	attention_weights = exponentiated_scores/np.sum(exponentiated_scores,axis=1,keepdims=True)
	output = np.matmul(attention_weights, V)
	return output,attention_weights