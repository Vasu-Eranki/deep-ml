import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    x_new = np.expand_dims(x_new, axis=0)

    d_k = W_K.shape[1]
    q = np.matmul(x_new,W_Q)
    k = np.matmul(x_new,W_K)
    v = np.matmul(x_new,W_V)
    K = V = None
    if not cache:
        K = k
        V = v 
    else:
        K,V = cache 
        K = np.vstack([K,k])
        V = np.vstack([V,v])
    cache = (K,V)

    scores = np.matmul(q, K.T)/(d_k)**0.5
    maximum_values = np.max(scores, axis=-1,keepdims=True)
    exponentiated_scores = np.exp(scores-maximum_values)
    softmax_scores = exponentiated_scores/np.sum(exponentiated_scores,axis=-1,keepdims=True)
    output = np.matmul(softmax_scores, V)
    return output.squeeze(),cache
        