import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    """
    Implement Xavier/Glorot weight initialization.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as numpy array
    """
    fan_in,fan_out = shape
    # Your code here
    np.random.seed(seed)
    if(mode=="uniform"):
        limit = (6/(fan_in+fan_out))**0.5
        initialization = np.random.uniform(-limit, limit, shape)
    else:
        variance = (2/(fan_in+fan_out))**0.5
        initialization= np.random.normal(0,variance,shape)
    return initialization
