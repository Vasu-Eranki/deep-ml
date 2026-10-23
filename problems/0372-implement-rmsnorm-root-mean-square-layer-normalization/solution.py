import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    
    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        g   : np.ndarray of shape (features,) - gain parameter
        eps : float - small constant for numerical stability
    
    Returns:
        np.ndarray of same shape as x
    """
    batch_size,features = x.shape
    sse = (np.sqrt(np.mean(np.power(x, 2),axis=-1))+eps)
    sse = np.expand_dims(sse, axis=-1)
    sse = np.hstack([sse]*features)
    return g*x/sse

    