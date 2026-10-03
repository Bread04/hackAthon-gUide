from common import *
import pandas as pd
from sliceline.slicefinder import Slicefinder
y_pred = (model.predict_proba(X_val)[:,1] > 0.15).astype(int)
X_binned = X_val.copy()
X_binned["age"] = pd.qcut(X_binned["age"], 5, duplicates="drop").astype(str)
X_binned = X_binned[["age"]].assign(discharge_to=df.loc[X_val.index,"discharge_to"].astype(str))
errors = (y_val != y_pred).astype(int)
sf = Slicefinder()
sf.fit(X_binned, errors)
print(sf.top_slices_)
