from common import *
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
pred_probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")
idx = find_label_issues(labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence")
print(len(idx), idx[:10])
