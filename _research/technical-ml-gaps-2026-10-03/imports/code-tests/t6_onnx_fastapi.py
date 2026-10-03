from common import *
import numpy as np
from skl2onnx import to_onnx
onx = to_onnx(model, X_train[:1].to_numpy().astype(np.float32))
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel
sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
app = FastAPI()
class Rows(BaseModel):
    rows: list[list[float]]
@app.post("/predict")
def predict(body: Rows):
    X_ = np.asarray(body.rows, dtype=np.float32)
    return {"label": sess.run([label_name], {inp: X_})[0].tolist()}
from fastapi.testclient import TestClient
c = TestClient(app)
r = c.post("/predict", json={"rows": X_val[:3].to_numpy().tolist()})
print(r.status_code, r.json(), "sklearn:", model.predict(X_val[:3]).tolist())
print("outputs:", [o.name for o in sess.get_outputs()])
