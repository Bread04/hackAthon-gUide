import numpy as np, pandas as pd
rng=np.random.default_rng(0); n=4000
X=rng.normal(size=(n,5)); w=rng.integers(0,2,n); p=1/(1+np.exp(-(X[:,0]+w*(X[:,1]>0)*1.0))); y=rng.binomial(1,p)
X_tr,X_te,w_tr,w_te,y_tr,y_te=X[:3000],X[3000:],w[:3000],w[3000:],y[:3000],y[3000:]
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from causalml.metrics.visualize import qini_score, auuc_score
m1 = HGB().fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
m0 = HGB().fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
uplift = m1.predict_proba(X_te)[:, 1] - m0.predict_proba(X_te)[:, 1]
df_eval = pd.DataFrame({'y': y_te, 'w': w_te, 't_learner': uplift})
print(qini_score(df_eval, outcome_col='y', treatment_col='w'))
print(auuc_score(df_eval, outcome_col='y', treatment_col='w'))
