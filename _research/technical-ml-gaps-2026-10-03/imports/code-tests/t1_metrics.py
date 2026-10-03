from common import *
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, balanced_accuracy_score,
    average_precision_score, roc_auc_score, log_loss, brier_score_loss,
    ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import LearningCurveDisplay
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
for name, m in [("dummy", dummy), ("model", model)]:
    p = m.predict_proba(X_val)[:, 1]
    yhat = m.predict(X_val)
    print(name,
          "bal_acc", balanced_accuracy_score(y_val, yhat),
          "AP", average_precision_score(y_val, p),
          "ROC-AUC", roc_auc_score(y_val, p),
          "logloss", log_loss(y_val, p),
          "brier", brier_score_loss(y_val, p))
print("prevalence", y_val.mean())
print(classification_report(y_val, model.predict(X_val), zero_division=0))
ConfusionMatrixDisplay.from_estimator(model, X_val, y_val)
CalibrationDisplay.from_estimator(model, X_val, y_val, n_bins=10)
LearningCurveDisplay.from_estimator(model, X_train, y_train)
import numpy as np
def precision_at_k(y_true, scores, k):
    top = np.argsort(-np.asarray(scores))[:k]
    return float(np.mean(np.asarray(y_true)[top]))
print("p@50", precision_at_k(y_val, model.predict_proba(X_val)[:,1], 50))
