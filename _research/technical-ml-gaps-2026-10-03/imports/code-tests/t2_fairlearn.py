from common import *
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
from sklearn.metrics import recall_score
y_true = y_val; y_pred = (model.predict_proba(X_val)[:,1] > 0.15).astype(int); sf_data = df.loc[X_val.index, "discharge_to"]
my_metrics = {'tpr': recall_score, 'fpr': false_positive_rate, 'sel': selection_rate, 'count': count}
mf = MetricFrame(metrics=my_metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sf_data)
print(mf.by_group); print(mf.difference(method='to_overall'))
