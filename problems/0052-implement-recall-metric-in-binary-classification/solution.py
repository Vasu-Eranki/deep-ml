import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    true_positive = np.sum(y_true[y_pred==1]==1)
    false_negative = np.sum(y_true[y_pred==0]==1)
    if (true_positive+false_negative)==0:
        return 0
    return true_positive/(true_positive+false_negative)
