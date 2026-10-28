import numpy as np

def attention(Q:np.ndarray,K:np.ndarray,V:np.ndarray)->np.ndarray:
    _,d_k = Q.shape 
    scores = np.matmul(Q, K.T)/(d_k**0.5)
    maximum_values = np.max(scores, axis=-1,keepdims=True)
    exponentiated_scores = np.exp(scores-maximum_values)
    softmax_scores = exponentiated_scores/np.sum(exponentiated_scores,axis=-1,keepdims=True)
    return np.matmul(softmax_scores, V)

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    K = np.matmul(X,W_key)
    V = np.matmul(X,W_value)
    cache = None 
    for W_Q in W_queries:
        Q = np.matmul(X,W_Q)
        attention_o = attention(Q,K,V)
        if cache is None:
            cache = attention_o
        else:
            cache = np.hstack([cache,attention_o])
            print(cache.shape,attention_o.shape)
    return np.matmul(cache,W_out)