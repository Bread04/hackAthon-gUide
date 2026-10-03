from common import *
import numpy as np, onnxruntime as rt
from skl2onnx import to_onnx
try:
    to_onnx(model, X_train[:1].astype(np.float32)); print("DataFrame input: OK")
except Exception as e: print("DataFrame input FAILS:", type(e).__name__, str(e)[:100])
onx = to_onnx(model, X_train[:1].to_numpy().astype(np.float32), options={"zipmap": False})
sess = rt.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
print("outputs:", [o.name for o in sess.get_outputs()]); out = sess.run([sess.get_outputs()[1].name], {sess.get_inputs()[0].name: X_val[:3].to_numpy().astype(np.float32)})[0]
print("onnx proba", np.round(out[:,1],4), "sklearn", np.round(model.predict_proba(X_val[:3])[:,1],4))
from sliceline.slicefinder import Slicefinder
import pandas as pd
y_pred = (model.predict_proba(X_val)[:,1] > 0.15).astype(int)
Xb = pd.DataFrame({"age": pd.qcut(X_val["age"],4).astype(str), "dest": df.loc[X_val.index,"discharge_to"].astype(str),
                   "prior": pd.cut(X_val["prior_admissions_12m"],[-1,0,1,10]).astype(str)})
sf = Slicefinder(alpha=0.95, k=3, max_l=2, min_sup=20).fit(Xb, (y_val != y_pred).astype(int))
print(sf.top_slices_)
