import numpy as np
def calculate_fpr(y_true,y_pred):
    false_positives = np.sum(y_pred[y_true==0]==1)
    true_negatives = np.sum(y_pred[y_true==0]==0)
    fpr =  false_positives/(false_positives+true_negatives)
    return fpr if not np.isnan(fpr) else 0
def calculate_tpr(y_true,y_pred):
    true_positives = np.sum(y_pred[y_true==1]==1)
    false_negatives = np.sum(y_pred[y_true==1]==0)
    tpr =  true_positives/(true_positives+false_negatives)
    return tpr if not np.isnan(tpr) else 0



def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    fpr = [0]
    tpr = [0]
    thresholds = np.unique(y_scores)[::-1]  # high → low
    y_scores_np = np.array(y_scores)
    y_true_np = np.array(y_true)
    for thresh in thresholds:
        y_pred_np = (y_scores_np>=thresh).astype(int)
        fpr_value = calculate_fpr(y_true_np,y_pred_np)
        tpr_value = calculate_tpr(y_true_np,y_pred_np)
        fpr.append(fpr_value)
        tpr.append(tpr_value)
    auc_values = []
    for index in range(1,len(fpr)):
        tpr_i = tpr[index]
        tpr_i1 = tpr[index-1]
        fpr_i = fpr[index]
        fpr_i1 = fpr[index-1]
        value = (fpr_i-fpr_i1)*(tpr_i+tpr_i1)/2
        auc_values.append(value)
    return sum(auc_values)
