import numpy as np, pandas as pd
rng=np.random.default_rng(0); n=2000
df=pd.DataFrame({"x":rng.uniform(0,50_000,n),"y":rng.uniform(0,50_000,n)})
X=np.c_[df.x,df.y,rng.normal(size=n)]; y=np.sin(df.x/8000)+np.cos(df.y/8000)+rng.normal(0,.1,n)
from sklearn.ensemble import RandomForestRegressor; model=RandomForestRegressor(n_estimators=50,random_state=0)
from sklearn.model_selection import GroupKFold, cross_val_score
block = 5_000
groups = (np.floor(df.x / block).astype(int).astype(str) + '_' +
          np.floor(df.y / block).astype(int).astype(str))
scores = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups)
print("block CV R2", scores.mean())
from verde import BlockKFold
print("verde folds", sum(1 for _ in BlockKFold(spacing=5_000, n_splits=5).split(df[["x","y"]].to_numpy())))
