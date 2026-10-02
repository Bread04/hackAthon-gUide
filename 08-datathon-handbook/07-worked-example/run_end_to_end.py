"""Run the handbook's runbook gates 2-6 end to end on sample-readmissions.csv.

    python run_end_to_end.py            # writes everything to ./outputs/

What it shows, gate by gate (see ../00-start-here/runbook.md):
  Gate 2  Lock evaluation: StratifiedGroupKFold by patient_id, folds saved, overlap asserted
  Gate 3  Data audit: rows, patients, target rate, missingness
  Leak    The planted post-outcome column is caught (score jump + importance share) and dropped
  Gate 4  Baseline ladder: dummy -> logistic regression -> LightGBM, all on the same folds
  Gate 5  One feature family at a time, each delta logged to feature_log.csv
  Gate 6  Decision threshold chosen on out-of-fold predictions by dollar value

The data is synthetic and the dollar figures are ASSUMPTIONS for illustration;
the point is the procedure, not the numbers.
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SEED = 42
TARGET, GROUP = "readmitted_30d", "patient_id"
LEAK_SUSPECT = "followup_call_outcome"
# Unit economics (ASSUMPTIONS, edit for your brief): a post-discharge nurse home visit costs $150;
# a prevented readmission saves $3,000 in penalties; the visit prevents 25% of readmissions.
# Break-even risk = 150 / (0.25 * 3000) = 0.20, so flag patients above roughly that risk.
COST_PER_FLAG, SAVING_PER_PREVENTED, EFFECTIVENESS = 150.0, 3000.0, 0.25


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def lgbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              random_state=SEED, verbose=-1)


def oof_predict(make_model, X, y, folds, cat_cols=()):
    oof = np.zeros(len(y))
    importances = []
    for k in sorted(folds.unique()):
        tr, va = folds != k, folds == k
        Xtr, Xva = X[tr].copy(), X[va].copy()
        for c in cat_cols:  # categories fixed from the training fold only
            cats = pd.CategoricalDtype(sorted(Xtr[c].dropna().unique()))
            Xtr[c], Xva[c] = Xtr[c].astype(cats), Xva[c].astype(cats)
        model = make_model().fit(Xtr, y[tr])
        oof[va.values] = model.predict_proba(Xva)[:, 1]
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
    imp = pd.Series(np.mean(importances, axis=0), index=X.columns) if importances else None
    return oof, imp


def logistic_for(num_cols, cat_cols=("discharge_to",)):
    """Logistic regression with preprocessing fitted inside each training fold."""
    def make():
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
             list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols))])
        return make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
    return make


def scores(y, p):
    return {"pr_auc": round(average_precision_score(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(HERE, "sample-readmissions.csv"), parse_dates=["admit_date"])
    y = df[TARGET]

    banner("Gate 3 · Data audit")
    audit = {
        "rows": len(df), "patients": df[GROUP].nunique(),
        "admissions_per_patient_max": int(df[GROUP].value_counts().max()),
        "target_rate": round(float(y.mean()), 4),
        "duplicate_admission_ids": int(df["admission_id"].duplicated().sum()),
        "missing_share": {c: round(float(v), 3) for c, v in df.isna().mean().items() if v > 0},
    }
    print(json.dumps(audit, indent=2))
    assert audit["duplicate_admission_ids"] == 0

    banner("Gate 2 · Lock evaluation (StratifiedGroupKFold by patient_id)")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = pd.Series(-1, index=df.index)
    for k, (_, va) in enumerate(sgkf.split(df, y, groups=df[GROUP])):
        folds.iloc[va] = k
    for k in range(5):  # no patient in both train and validation
        assert not set(df.loc[folds == k, GROUP]) & set(df.loc[folds != k, GROUP])
    pd.DataFrame({"admission_id": df["admission_id"], "fold": folds}).to_csv(
        os.path.join(OUT, "folds.csv"), index=False)
    print("5 folds saved to outputs/folds.csv; no patient appears in two folds.")
    print("Positives per fold:", df.groupby(folds)[TARGET].sum().to_dict())

    base_num = ["age", "chronic_conditions", "length_of_stay", "n_medications",
                "prior_admissions_12m", "creatinine"]
    cats = ["discharge_to"]

    banner("Leakage check · does any column look too good?")
    X_all = df[base_num + cats + [LEAK_SUSPECT]]
    oof_leak, imp = oof_predict(lgbm, X_all, y, folds, cat_cols=cats + [LEAK_SUSPECT])
    share = (imp / imp.sum()).sort_values(ascending=False)
    oof_clean, _ = oof_predict(lgbm, df[base_num + cats], y, folds, cat_cols=cats)
    s_leak, s_clean = scores(y, oof_leak), scores(y, oof_clean)
    print(f"With {LEAK_SUSPECT}: {s_leak}   without: {s_clean}")
    print(f"Top importance share: {share.index[0]} = {share.iloc[0]:.0%}")
    jump = s_leak["pr_auc"] - s_clean["pr_auc"]
    flagged = jump > 0.15 or (share.index[0] == LEAK_SUSPECT and share.iloc[0] > 0.30)
    print(("LEAK SUSPECTED" if flagged else "no leak signal") +
          f": PR-AUC jump {jump:+.3f}. '{LEAK_SUSPECT}' is recorded after the outcome, so it is dropped.")

    banner("Why the split matters · random KFold vs grouped folds (same model)")
    rand = pd.Series(-1, index=df.index)
    for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(df)):
        rand.iloc[va] = k
    oof_rand, _ = oof_predict(lgbm, df[base_num + cats], y, rand, cat_cols=cats)
    print(f"Random KFold: {scores(y, oof_rand)}   Grouped: {s_clean}")
    print("Report the grouped score: new patients are what the model will see.")

    banner("Gate 4 · Baseline ladder (all on the locked folds)")
    ladder = {"dummy (predict the base rate)": scores(y, np.full(len(y), y.mean()))}
    oof_lr, _ = oof_predict(logistic_for(base_num), df[base_num + cats], y, folds)
    ladder["logistic regression"] = scores(y, oof_lr)
    ladder["LightGBM (default-ish)"] = s_clean
    for name, s in ladder.items():
        print(f"{name:32} PR-AUC {s['pr_auc']:.4f}  ROC-AUC {s['roc_auc']:.4f}")
    assert ladder["LightGBM (default-ish)"]["pr_auc"] > ladder["dummy (predict the base rate)"]["pr_auc"]
    use_logistic = ladder["logistic regression"]["pr_auc"] >= ladder["LightGBM (default-ish)"]["pr_auc"]
    winner = "logistic regression" if use_logistic else "LightGBM (default-ish)"
    print(f"Winner on the locked folds: {winner}. The simpler model is kept unless the complex one beats it.")

    def run(frame):
        num = [c for c in frame.columns if c not in cats]
        make = logistic_for(num) if use_logistic else lgbm
        return oof_predict(make, frame, y, folds, cat_cols=() if use_logistic else cats)[0]

    banner("Gate 5 · One feature family at a time (logged to feature_log.csv)")
    feats = df[base_num + cats].copy()
    best_oof = oof_lr if use_logistic else oof_clean
    best = scores(y, best_oof)["pr_auc"]
    log = [{"step": f"baseline ({winner})", "family": "-", "pr_auc": best, "delta": 0.0, "kept": True}]
    families = {
        "missingness flag": lambda d: d.assign(creatinine_missing=d["creatinine"].isna().astype(int)),
        "ratios": lambda d: d.assign(meds_per_condition=d["n_medications"] / (d["chronic_conditions"] + 1),
                                     stay_x_prior=d["length_of_stay"] * d["prior_admissions_12m"]),
        "patient history (past admissions only)": lambda d: d.assign(
            n_prev_admissions=df.groupby(GROUP)["admit_date"].rank(method="first").astype(int) - 1),
    }
    for fam, add in families.items():
        trial = add(feats)
        oof_t = run(trial)
        s = scores(y, oof_t)
        delta = round(s["pr_auc"] - best, 4)
        kept = delta > 0.002  # keep only real gains, not noise
        log.append({"step": fam, "family": fam, "pr_auc": s["pr_auc"], "delta": delta, "kept": kept})
        print(f"{fam:42} PR-AUC {s['pr_auc']:.4f} ({delta:+.4f}) -> {'keep' if kept else 'drop'}")
        if kept:
            feats, best, best_oof = trial, s["pr_auc"], oof_t
    pd.DataFrame(log).to_csv(os.path.join(OUT, "feature_log.csv"), index=False)
    pd.DataFrame({"admission_id": df["admission_id"], "fold": folds, "oof": best_oof, TARGET: y}).to_csv(
        os.path.join(OUT, "final_oof.csv"), index=False)

    banner("Gate 6 · Threshold by dollar value (on out-of-fold predictions)")

    def value(t):
        flag = best_oof >= t
        tp = int((flag & (y == 1)).sum())
        return tp * EFFECTIVENESS * SAVING_PER_PREVENTED - flag.sum() * COST_PER_FLAG, int(flag.sum()), tp

    grid = np.round(np.arange(0.02, 0.9, 0.01), 2)
    vals = {t: value(t) for t in grid}
    t_best = max(vals, key=lambda t: vals[t][0])
    v_best, v_half = vals[t_best], value(0.5)
    print(f"Assumptions: ${COST_PER_FLAG:.0f} per home visit, ${SAVING_PER_PREVENTED:,.0f} per prevented "
          f"readmission, {EFFECTIVENESS:.0%} effectiveness (edit at the top of the script).")
    print(f"Best threshold {t_best:.2f}: flag {v_best[1]} of {len(y)}, catch {v_best[2]} readmissions, "
          f"net ${v_best[0]:,.0f}")
    print(f"Default 0.50:          flag {v_half[1]}, catch {v_half[2]}, net ${v_half[0]:,.0f}")
    breakeven = COST_PER_FLAG / (EFFECTIVENESS * SAVING_PER_PREVENTED)
    if use_logistic and abs(t_best - breakeven) > 0.1:
        print(f"Note: the break-even risk is {breakeven:.2f}, but the best cut-off is {t_best:.2f} because "
              "class_weight='balanced' inflates probabilities. Calibrate (CalibratedClassifierCV) before "
              "quoting probabilities to judges; always pick the threshold on OOF predictions.")
    json.dump({"threshold": float(t_best), "net_value": v_best[0], "assumptions": {
        "cost_per_flag": COST_PER_FLAG, "saving_per_prevented": SAVING_PER_PREVENTED,
        "effectiveness": EFFECTIVENESS}}, open(os.path.join(OUT, "threshold.json"), "w"), indent=2)

    banner("Rules of evidence · every pitch number has a file")
    evidence = pd.DataFrame([
        {"claim": f"Grouped OOF PR-AUC = {best:.3f} with {winner} (base rate {y.mean():.3f})", "file": "outputs/final_oof.csv"},
        {"claim": "Each feature family's gain", "file": "outputs/feature_log.csv"},
        {"claim": f"Net value ${v_best[0]:,.0f} at threshold {t_best:.2f} (assumed costs)", "file": "outputs/threshold.json"},
        {"claim": "No patient in two folds", "file": "outputs/folds.csv"},
    ])
    print(evidence.to_string(index=False))
    json.dump(audit, open(os.path.join(OUT, "data_audit.json"), "w"), indent=2)
    print(f"\nDone. Outputs in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
