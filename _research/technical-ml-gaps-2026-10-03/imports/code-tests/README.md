# Code tests for research run 11 (2026-10-03)

Scripts used to run the report's code on the guide's sample data
(`08-datathon-handbook/07-worked-example/sample-readmissions.csv`).
Environment: Python 3.12.3, the pinned `PROJECT_TEMPLATE/requirements.txt`, plus
fairlearn 0.14.0, sliceline 0.3.0, cleanlab 2.9.0, evidently 0.7.23, skl2onnx 1.20.0,
onnx 1.23.1, onnxruntime 1.30.0, fastapi 0.142.2, httpx 0.28.1, Faker 40.40.0.

| Script | Tests | Result |
| --- | --- | --- |
| `t1_metrics.py` | Metrics, displays, precision@k | Runs |
| `t2_fairlearn.py` | MetricFrame by segment | Runs |
| `t3_sliceline.py` | Slicefinder with defaults | Runs but finds **no slices** |
| `t9.py` | Slicefinder with `min_sup`, ONNX `zipmap=False`, DataFrame input | Slices found; output names are `label` / `probabilities`; ONNX probabilities match scikit-learn |
| `t4_cleanlab.py` | `find_label_issues(..., return_indices_ranked_by="self_confidence")` | Runs |
| `t5_evidently.py` | Data drift report | Runs |
| `t6_onnx_fastapi.py` | scikit-learn → ONNX → FastAPI `TestClient` | 200 OK, labels match scikit-learn |
| `t7_llmlabels.py` | LLM-vs-human label agreement, Faker | Runs (toy labels) |
| `t8_jsonl.py` | JSONL run logger | Runs; read back with pandas |

Not run (blocked downloads in the sandbox): PyTorch, timm, transformers, SetFit, PEFT,
Ultralytics, transformers.js, Hugging Face Inference Providers, Trackio, MLflow, DVC.
