"""Run the datathon handbook's Python scripts end to end on synthetic data.

Usage (from the repo root, inside an env built from
08-datathon-handbook/PROJECT_TEMPLATE/requirements.txt):

    python tools/smoke_test.py

Works on a temporary copy, so nothing is written into the repo.
Exit code 0 = every check passed.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HB = os.path.join(ROOT, "08-datathon-handbook")
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, "PASS", ""))
    except Exception as e:  # report every failure, keep going
        results.append((name, "FAIL", str(e)[:300]))


def run(cmd, cwd, stdin=None, timeout=900):
    p = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    return p.stdout


def shap_on_tree_models():
    import numpy as np
    import shap
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    rng = np.random.default_rng(0)
    X = rng.random((300, 5))
    y = (X[:, 0] > 0.5).astype(int)
    for m in (lgb.LGBMClassifier(n_estimators=20, verbose=-1),
              xgb.XGBClassifier(n_estimators=20),
              cb.CatBoostClassifier(iterations=20, verbose=0, allow_writing_files=False)):
        shap.TreeExplainer(m.fit(X, y)).shap_values(X[:5])


with tempfile.TemporaryDirectory() as tmp:
    for d in ("00-start-here", "03-modeling", "04-solutions-and-ui"):
        shutil.copytree(os.path.join(HB, d), os.path.join(tmp, d))
    py = sys.executable
    check("SHAP TreeExplainer on LightGBM, XGBoost, CatBoost", shap_on_tree_models)
    check("03-modeling/baseline-pipeline.py",
          lambda: run([py, "baseline-pipeline.py"], os.path.join(tmp, "03-modeling")))
    check("00-start-here/quickstart-1-click.py",
          lambda: run([py, "quickstart-1-click.py"], os.path.join(tmp, "00-start-here"), stdin="n\n"))

    def app():
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(tmp, "04-solutions-and-ui", "streamlit-app-template.py"),
                               default_timeout=180).run()
        if at.exception:
            raise RuntimeError(str([e.value for e in at.exception]))
    check("04-solutions-and-ui/streamlit-app-template.py (AppTest)", app)

    try:
        import autogluon.tabular  # noqa: F401
        check("03-modeling/automl-autogluon.py",
              lambda: run([py, "automl-autogluon.py"], os.path.join(tmp, "03-modeling"), timeout=1800))
    except ImportError:
        results.append(("03-modeling/automl-autogluon.py", "SKIP", "autogluon not installed (optional)"))

print(f"Python {sys.version.split()[0]}")
for name, status, note in results:
    print(f"{status:4}  {name}" + (f"  -> {note}" if note else ""))
sys.exit(1 if any(s == "FAIL" for _, s, _ in results) else 0)
