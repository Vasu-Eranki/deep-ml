import numpy as np
def precision(y_true, y_pred):
	# Your code here
	true_positives = np.sum(y_true[y_pred==1]==1)
	false_positives =np.sum(y_true[y_pred==1]==0)
	if(true_positives+false_positives)==0:
		return 0
	return true_positives/(true_positives+false_positives)
