import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = np.matmul(X, W_q)
    K = np.matmul(X, W_k)
    V = np.matmul(X, W_v)
    return Q,K,V
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    _,d_k = Q.shape 
    scaledscores = np.matmul(Q, K.T)/(d_k)**0.5
    maximum_value = np.max(scaledscores, axis=-1,keepdims=True)
    exponentiated_scores = np.exp(scaledscores-maximum_value)
    softmax_scores = exponentiated_scores/np.sum(exponentiated_scores,axis=-1,keepdims=True)
    return np.matmul(softmax_scores, V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    multiQ = np.split(Q, n_heads,axis=-1)
    multiK = np.split(K,n_heads,axis=-1)
    multiV = np.split(V,n_heads,axis=-1)
    results = []
    for Q,K,V in zip(multiQ,multiK,multiV):
        attention = self_attention(Q,K,V)
        results.append(attention)
    
    return np.concatenate(results,axis=-1)