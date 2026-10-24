import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    np.random.seed(seed)
    match (mode,distribution):
        case ("fan_in","normal"):
            stddev = (2/n_in)**0.5
            return np.random.normal(0,stddev,(n_in,n_out))
        case ("fan_out","normal"):
            stddev = (2/n_out)**0.5
            return np.random.normal(0,stddev,(n_in,n_out))
        case ("fan_in","uniform"):
            limit = (6/n_in)**0.5
            return np.random.uniform(-limit,limit,(n_in,n_out))
        case ("fan_out","uniform"):
            limit = (6/n_out)**0.5
            return np.random.uniform(-limit,limit,(n_in,n_out))
