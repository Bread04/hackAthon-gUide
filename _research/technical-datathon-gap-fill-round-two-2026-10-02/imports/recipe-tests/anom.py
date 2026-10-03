import numpy as np
rng=np.random.default_rng(0)
X_train=rng.normal(size=(1000,6)); X_test=np.vstack([rng.normal(size=(950,6)),rng.normal(4,1,size=(50,6))]); y_test=np.r_[np.zeros(950),np.ones(50)]
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
for name, det in [('ECOD', ECOD()), ('IForest', IForest(n_estimators=300, random_state=0))]:
    det.fit(X_train)
    s = det.decision_function(X_test)
    print(name, 'PR-AUC', average_precision_score(y_test, s), 'base rate', y_test.mean())
