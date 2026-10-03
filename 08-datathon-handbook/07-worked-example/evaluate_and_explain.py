"""Metrics, error analysis and a one-line experiment log, on sample-readmissions.csv.

    python evaluate_and_explain.py

Companion to ../03-modeling/ml-fundamentals-and-metrics.md and
../03-modeling/error-analysis-and-tracking.md. Uses the same grouped folds as
run_end_to_end.py, so every number here is out-of-fold (never seen in training).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SEED = 42
DATA = os.path.join(HERE, "sample-readmissions.csv")
NUM = ["age", "chronic_conditions", "length_of_stay", "n_medications", "prior_admissions_12m", "creatinine"]
CAT = ["discharge_to"]
THRESHOLD = 0.5  # the decision cut-off we explain below; choose yours by $ (runbook Gate 6)


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def make_model(calibrated=False):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    # class_weight=None keeps probabilities honest; "balanced" would inflate them
    return make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight=None if calibrated else "balanced"))


def oof(df, y, folds, calibrated=False):
    p = np.zeros(len(y))
    for k in range(5):
        tr, va = folds != k, folds == k
        p[va] = make_model(calibrated).fit(df[tr], y[tr]).predict_proba(df[va])[:, 1]
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    y = df["readmitted_30d"].to_numpy()
    folds = np.full(len(y), -1)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                                .split(df, y, groups=df["patient_id"])):
        folds[va] = k
    X = df[NUM + CAT]
    p = oof(X, y, folds)
    base = y.mean()

    banner("1 · Confusion matrix at threshold %.2f (rows = truth, columns = prediction)" % THRESHOLD)
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"                predicted no   predicted yes\n actually no    {tn:>12}   {fp:>13}\n actually yes   {fn:>12}   {tp:>13}")
    print(f"\nPrecision = TP/(TP+FP) = {tp}/{tp + fp} = {precision_score(y, pred):.3f}  "
          "-> of the patients we flag, how many really get readmitted")
    print(f"Recall    = TP/(TP+FN) = {tp}/{tp + fn} = {recall_score(y, pred):.3f}  "
          "-> of the patients who get readmitted, how many we catch")
    print(f"F1 = {f1_score(y, pred):.3f} · balanced accuracy = {balanced_accuracy_score(y, pred):.3f}")
    print(f"Plain accuracy = {(tp + tn) / len(y):.3f}, but predicting 'nobody' scores {1 - base:.3f}: "
          "accuracy hides the minority class.")

    banner("2 · Threshold-free scores, each next to its 'no skill' baseline")
    print(f"PR-AUC (average precision) = {average_precision_score(y, p):.3f}   no-skill = positive rate = {base:.3f}")
    print(f"ROC-AUC                    = {roc_auc_score(y, p):.3f}   no-skill = 0.500")
    print("Report both with their baselines: they answer different questions (finding the rare positives vs ranking overall).")

    banner("3 · Are the probabilities honest? (calibration)")
    p_cal = oof(X, y, folds, calibrated=True)
    for name, q in [("class_weight='balanced'", p), ("no class weights", p_cal)]:
        frac, mean_pred = calibration_curve(y, q, n_bins=5, strategy="quantile")
        print(f"{name:24} Brier {brier_score_loss(y, q):.3f} · log loss {log_loss(y, q):.3f} · "
              f"mean predicted {q.mean():.3f} vs actual rate {base:.3f}")
        print("   bins (predicted -> observed): " +
              ", ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(mean_pred, frac)))
    print("Balanced weights help ranking but inflate probabilities; quote probabilities from a calibrated model.")

    banner("4 · Learning curve: would more data help?")
    sizes, tr_s, va_s = learning_curve(make_model(), X, y, groups=df["patient_id"],
                                       cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                                       train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring="average_precision")
    for n, a, b in zip(sizes, tr_s.mean(1), va_s.mean(1)):
        print(f"train rows {n:>5}: train PR-AUC {a:.3f} · validation PR-AUC {b:.3f} · gap {a - b:+.3f}")
    v, gap = va_s.mean(1), tr_s.mean(1) - va_s.mean(1)
    still_rising = v[-1] - v[-2] > 0.005
    print(("Validation score is still rising at full size -> more (or more varied) data would likely help."
           if still_rising else "Validation score has flattened -> more rows won't help much; better features might.")
          + (" Train/validation gap is small, so the model is not badly overfitting." if gap[-1] < 0.05
             else " Large train/validation gap -> overfitting: simplify or regularise."))

    banner("5 · Error analysis: where does the model fail? (slices of out-of-fold predictions)")
    e = df.assign(p=p, y=y, age_band=pd.cut(df["age"], [0, 40, 65, 80, 120], labels=["<40", "40-64", "65-79", "80+"]),
                  creatinine_missing=df["creatinine"].isna())
    rows = []
    for col in ["discharge_to", "age_band", "creatinine_missing"]:
        for val, g in e.groupby(col, observed=True):
            if g["y"].nunique() < 2 or len(g) < 50:
                continue
            rows.append({"slice": f"{col}={val}", "n": len(g), "positive_rate": round(g["y"].mean(), 3),
                         "pr_auc": round(average_precision_score(g["y"], g["p"]), 3),
                         "lift_over_base": round(average_precision_score(g["y"], g["p"]) / g["y"].mean(), 2)})
    slices = pd.DataFrame(rows).sort_values("lift_over_base")
    print(slices.to_string(index=False))
    print("Weakest slices (lowest lift) are where to look for new features or to state a limitation in the pitch.")
    worst = e.assign(err=np.abs(e["y"] - e["p"])).sort_values("err", ascending=False)
    worst.head(20)[["admission_id", "patient_id", "y", "p"] + NUM + CAT].to_csv(os.path.join(OUT, "worst_errors.csv"), index=False)
    missed = worst[(worst["y"] == 1)].head(20)
    print(f"\nTop-20 missed readmissions vs all readmissions (medians):")
    print(pd.DataFrame({"missed": missed[NUM].median(), "all_readmitted": e[e["y"] == 1][NUM].median()}).round(2).to_string())
    looks_low_risk = (missed[NUM].median() < e[e["y"] == 1][NUM].median()).sum()
    if looks_low_risk >= len(NUM) - 1:
        print("The missed readmissions look low-risk on almost every recorded feature: the driver is probably "
              "something the data does not measure. Say so in the pitch, and ask what extra data would capture it.")
    print("Saved the 20 worst errors to outputs/worst_errors.csv: read them by hand and tag the patterns.")

    banner("6 · Experiment log: one line per run (outputs/experiments.jsonl)")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip() or "no-git"
    except OSError:
        commit = "no-git"
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit,
        "data_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest()[:12],
        "model": "logistic_regression(class_weight=balanced)", "features": NUM + CAT,
        "cv": "StratifiedGroupKFold(5, patient_id, seed=42)", "seed": SEED,
        "pr_auc_oof": round(average_precision_score(y, p), 4), "roc_auc_oof": round(roc_auc_score(y, p), 4),
        "pr_auc_per_fold": [round(average_precision_score(y[folds == k], p[folds == k]), 4) for k in range(5)],
        "note": "baseline for metrics page",
    }
    with open(os.path.join(OUT, "experiments.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    print("Commit experiments.jsonl with the code: every pitch number then points to a run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
