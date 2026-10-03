"""LightGBM imbalance lab: what each imbalance setting does to ranking AND to probabilities,
how to fix probabilities, and how to pick a threshold from false-positive / false-negative costs.

    python lightgbm_imbalance_lab.py              # sample data as-is (~11% positive)
    python lightgbm_imbalance_lab.py --rare       # keep ~3% positives to make imbalance harsher

Companion to ../03-modeling/lightgbm-imbalance-and-tuning.md. All numbers are out-of-fold on grouped
folds (no patient in two folds). The data is synthetic: the point is the *pattern*, not the scores.
"""
import argparse
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
NUM = ["age", "chronic_conditions", "length_of_stay", "n_medications", "prior_admissions_12m", "creatinine"]
CAT = ["discharge_to"]
# Decision costs (ASSUMPTIONS, same story as run_end_to_end.py):
# flagging a patient = $150 nurse visit (paid for every flag, TP or FP);
# a flagged patient who would have been readmitted saves 25% x $3,000 = $750 on average.
COST_FP = 150.0                  # money wasted on a flag that was not needed
BENEFIT_TP = 0.25 * 3000 - 150   # net gain from a correct flag = 600
COST_FN = 0.25 * 3000            # value lost by missing a readmission we could have prevented = 750


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(rare):
    df = pd.read_csv(os.path.join(HERE, "sample-readmissions.csv"))
    if rare:  # drop most positives to simulate a ~3% positive rate
        rng = np.random.default_rng(SEED)
        pos = df.index[df["readmitted_30d"] == 1]
        drop = rng.choice(pos, size=int(len(pos) * 0.75), replace=False)
        df = df.drop(index=drop).reset_index(drop=True)
    df["discharge_to"] = df["discharge_to"].astype("category")
    return df


def model(**kw):
    base = dict(n_jobs=min(4, os.cpu_count() or 1), n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=SEED, verbose=-1)
    base.update(kw)
    return lgb.LGBMClassifier(**base)


def oof(df, y, folds, make, undersample=None, calibrate=None):
    """Out-of-fold probabilities. undersample=ratio keeps that many negatives per positive in TRAIN only.
    calibrate='sigmoid'|'isotonic' calibrates on a held-out group split inside the training fold."""
    X = df[NUM + CAT]
    p = np.zeros(len(y))
    for tr, va in folds:
        tr_idx = np.asarray(tr)
        if calibrate:
            inner = GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
            fit_i, cal_i = next(inner.split(tr_idx, groups=df["patient_id"].iloc[tr_idx]))
            fit_idx, cal_idx = tr_idx[fit_i], tr_idx[cal_i]
        else:
            fit_idx = tr_idx
        if undersample:
            rng = np.random.default_rng(SEED)
            pos = fit_idx[y[fit_idx] == 1]
            neg = fit_idx[y[fit_idx] == 0]
            neg = rng.choice(neg, size=min(len(neg), int(len(pos) * undersample)), replace=False)
            fit_idx = np.concatenate([pos, neg])
        m = make().fit(X.iloc[fit_idx], y[fit_idx])
        if calibrate:
            m = CalibratedClassifierCV(FrozenEstimator(m), method=calibrate).fit(X.iloc[cal_idx], y[cal_idx])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


def crossfit_platt(p_oof, y, folds):
    """Calibrate out-of-fold scores without leakage: for each fold, fit Platt scaling (logistic regression
    on the log-odds) using the OTHER folds' out-of-fold scores, then apply it to this fold. Keeps the full
    training data for the model and keeps the ranking (the map is monotone)."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p_oof, 1e-6, 1 - 1e-6) / (1 - np.clip(p_oof, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    out = np.zeros(len(y))
    for tr, va in folds:
        out[va] = LogisticRegression(C=1e6).fit(z[tr], y[tr]).predict_proba(z[va])[:, 1]
    return out


def quality(y, p):
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p), "log loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "mean p": p.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rare", action="store_true")
    args = ap.parse_args()
    df = load(args.rare)
    y = df["readmitted_30d"].to_numpy()
    pos_rate = y.mean()
    spw = (1 - pos_rate) / pos_rate
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, df["patient_id"]))

    banner(f"1 · Imbalance settings: ranking vs probability quality (positive rate {pos_rate:.3f}, "
           f"neg/pos = {spw:.1f})")
    runs = {
        "no weighting (default)": oof(df, y, folds, lambda: model()),
        f"scale_pos_weight={spw:.1f}": oof(df, y, folds, lambda: model(scale_pos_weight=spw)),
        "is_unbalance=True": oof(df, y, folds, lambda: model(is_unbalance=True)),
        "class_weight='balanced'": oof(df, y, folds, lambda: model(class_weight="balanced")),
        "undersample negatives 1:1": oof(df, y, folds, lambda: model(), undersample=1.0),
    }
    rows = [{"setting": k, **quality(y, v)} for k, v in runs.items()]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Actual positive rate: {pos_rate:.3f}. Compare the 'mean p' column: weighting and undersampling "
          "push predicted probabilities far above reality, while ranking (PR-AUC, ROC-AUC) changes much less.")

    banner("2 · Fixing the probabilities")
    p_w = runs[f"scale_pos_weight={spw:.1f}"]
    # Prior-shift correction: undo a positive-class weight w by dividing the odds by w.
    p_corr = p_w / (p_w + (1 - p_w) * spw)
    fixes = {
        "weighted, uncorrected": p_w,
        "weighted + odds correction (divide odds by w)": p_corr,
        "weighted + cross-fitted Platt on OOF scores": crossfit_platt(p_w, y, folds),
        "unweighted + cross-fitted Platt on OOF scores": crossfit_platt(runs["no weighting (default)"], y, folds),
        "unweighted + sigmoid (Platt) calibration": oof(df, y, folds, lambda: model(), calibrate="sigmoid"),
        "unweighted + isotonic calibration": oof(df, y, folds, lambda: model(), calibrate="isotonic"),
        "unweighted, no calibration": runs["no weighting (default)"],
    }
    rows = [{"probabilities": k, **quality(y, v)} for k, v in fixes.items()]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for name in ("weighted, uncorrected", "weighted + odds correction (divide odds by w)",
                 "weighted + cross-fitted Platt on OOF scores"):
        frac, mean_pred = calibration_curve(y, fixes[name], n_bins=5, strategy="quantile")
        print(f"{name}: predicted -> observed  " + ", ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(mean_pred, frac)))
    over = p_corr.mean() < 0.8 * pos_rate
    print("Lower Brier / log loss = more honest probabilities.")
    if over:
        print(f"The textbook odds correction OVER-corrected here (mean p {p_corr.mean():.3f} vs actual {pos_rate:.3f}): "
              "it is exact only for an ideal model. Fit a calibrator on held-out predictions instead.")
    print("Holding out a calibration slice inside each training fold costs training data (watch PR-AUC drop); "
          "cross-fitting Platt on out-of-fold scores keeps all the data and the ranking.")

    banner("3 · False positives vs false negatives: choosing the threshold from costs")
    p = fixes["weighted + cross-fitted Platt on OOF scores"]
    p_star = COST_FP / (COST_FP + BENEFIT_TP + 0)  # flag when p*BENEFIT_TP > (1-p)*COST_FP
    print(f"Assumed: a flag costs ${COST_FP:.0f}; a correct flag nets ${BENEFIT_TP:.0f}; a miss forgoes ${COST_FN:.0f}.")
    print(f"Decision rule for CALIBRATED probabilities: flag if p > COST_FP / (COST_FP + BENEFIT_TP) = {p_star:.3f}")

    def table(probs, thresholds):
        out = []
        for t in thresholds:
            flag = probs >= t
            tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
            fn = int((~flag & (y == 1)).sum())
            out.append({"threshold": t, "flagged": int(flag.sum()), "TP": tp, "FP": fp, "FN": fn,
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                        "net $": tp * BENEFIT_TP - fp * COST_FP})
        return pd.DataFrame(out)

    grid = sorted({round(p_star, 3), 0.05, 0.1, 0.3, 0.5})
    print(table(p, grid).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = table(p, np.round(np.arange(0.02, 0.9, 0.01), 2)).sort_values("net $").iloc[-1]
    print(f"Best threshold found by scanning OOF predictions: {best['threshold']:.2f} (net ${best['net $']:,.0f}); "
          f"theory says {p_star:.3f}. They agree when probabilities are calibrated.")
    t_w = table(p_w, [p_star]).iloc[0]
    print(f"Same rule applied to the UNCORRECTED weighted probabilities flags {int(t_w['flagged'])} patients "
          f"(net ${t_w['net $']:,.0f}): a theory-based threshold only works on calibrated probabilities.")

    banner("4 · Fixed budget instead of costs: 'we can call 100 patients a week'")
    k = 100 if not args.rare else 40
    top = np.argsort(-p)[:k]
    print(f"Top {k} by risk: precision {y[top].mean():.3f} (vs base rate {pos_rate:.3f}, lift {y[top].mean() / pos_rate:.1f}x), "
          f"catching {y[top].sum()} of {y.sum()} readmissions (recall {y[top].sum() / y.sum():.3f}).")
    print("With a capacity limit, rank and take the top k: ranking quality (PR-AUC) matters, calibration less so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
