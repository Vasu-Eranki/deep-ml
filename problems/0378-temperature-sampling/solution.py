import numpy as np
def softmax(logits:np.ndarray)->np.ndarray:
	maximum_value = np.max(logits, axis=-1,keepdims=True)
	scaled_scores = np.exp(logits-maximum_value)
	softmax_scores = scaled_scores/np.sum(scaled_scores,axis=-1,keepdims=True)
	return softmax_scores

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
	if temperature>0:
		scaled_logits = logits/temperature
		return softmax(scaled_logits)
	else:
		maximum_value_idx = np.argmax(logits,axis=-1)
		scores = np.zeros_like(logits)
		scores[maximum_value_idx]=1.0
		return scores