"""Train every classic ML model on the sample data and compare them fairly.

    python model_zoo.py            # ~1-2 minutes on a laptop

Companion to ../03-modeling/ml-models-explained.md. Every model uses the SAME
grouped folds (by patient_id) and the SAME preprocessing fitted inside each fold,
so the comparison is fair. Prints:
  A. classification (will this admission be readmitted?)  -> PR-AUC, ROC-AUC, fit time
  B. regression (how many medications at discharge?)     -> MAE, RMSE, R2
  C. how to read linear and logistic regression coefficients
  D. unsupervised models: clustering, PCA, anomaly detection
The data is synthetic; rankings on YOUR data can differ. That is the point of
running a model zoo on your own folds.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, mean_absolute_error, r2_score,
                             roc_auc_score, root_mean_squared_error, silhouette_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
NUM = ["age", "chronic_conditions", "length_of_stay", "n_medications", "prior_admissions_12m", "creatinine"]
CAT = ["discharge_to"]
REG_TARGET = "n_medications"                       # a count that depends on chronic conditions
NUM_REG = [c for c in NUM if c != REG_TARGET]      # never use the target as a feature


def prep(scale=True, num_cols=None):
    """Impute + (optionally) scale numbers, one-hot categories; fitted per training fold."""
    num = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([("num", make_pipeline(*num), num_cols or NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def classifiers():
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    return {
        "Dummy (predict base rate)": (DummyClassifier(strategy="prior"), True),
        "Logistic regression": (LogisticRegression(max_iter=2000), True),
        "Logistic regression (L1, C=0.1)": (LogisticRegression(l1_ratio=1.0, C=0.1, solver="saga", max_iter=5000), True),
        "Naive Bayes (Gaussian)": (GaussianNB(), True),
        "k-nearest neighbours (k=50)": (KNeighborsClassifier(n_neighbors=50), True),
        "SVM (RBF kernel)": (SVC(C=1.0, probability=True, random_state=SEED), True),
        "Decision tree (depth 4)": (DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=SEED), False),
        "Decision tree (no depth limit)": (DecisionTreeClassifier(random_state=SEED), False),
        "Random forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED), False),
        "Extra trees": (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED), False),
        "AdaBoost": (AdaBoostClassifier(n_estimators=200, random_state=SEED), False),
        "HistGradientBoosting (sklearn)": (HistGradientBoostingClassifier(random_state=SEED), False),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1, random_state=SEED), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=SEED), False),
        "CatBoost": (cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, verbose=0,
                                           random_seed=SEED, allow_writing_files=False), False),
        "Neural net (MLP 32-16)": (MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
                                                 max_iter=500, random_state=SEED), True),
    }


def regressors():
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    import lightgbm as lgb
    return {
        "Dummy (predict the mean)": (DummyRegressor(), True),
        "Linear regression": (LinearRegression(), True),
        "Ridge (L2, alpha=10)": (Ridge(alpha=10.0), True),
        "Lasso (L1, alpha=0.05)": (Lasso(alpha=0.05), True),
        "Poisson regression (GLM)": (PoissonRegressor(alpha=1e-3, max_iter=1000), True),
        "k-nearest neighbours (k=30)": (KNeighborsRegressor(n_neighbors=30), True),
        "Decision tree (depth 4)": (DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED), False),
        "Random forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED), False),
        "HistGradientBoosting (sklearn)": (HistGradientBoostingRegressor(random_state=SEED), False),
        "LightGBM": (lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1, random_state=SEED), False),
    }


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    df = pd.read_csv(os.path.join(HERE, "sample-readmissions.csv"))
    X, groups = df[NUM + CAT], df["patient_id"]

    banner("A · Classification: readmitted within 30 days? (grouped 5-fold OOF, positive rate %.3f)"
           % df["readmitted_30d"].mean())
    y = df["readmitted_30d"].to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    rows = []
    for name, (est, scale) in classifiers().items():
        oof, t0 = np.zeros(len(y)), time.time()
        for tr, va in folds:
            pipe = make_pipeline(prep(scale), est).fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        rows.append({"model": name, "PR-AUC": average_precision_score(y, oof),
                     "ROC-AUC": roc_auc_score(y, oof), "fit+predict s": time.time() - t0})
    res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("No-skill PR-AUC = positive rate; no-skill ROC-AUC = 0.500.")
    print("Caution: this synthetic data was GENERATED from a linear (logistic) formula, so linear models are\n"
          "expected to win here. On real tabular data with interactions, gradient-boosted trees usually lead.\n"
          "Run the zoo on your own folds instead of trusting any ranking, including this one.")

    banner("B · Regression: number of medications at discharge (grouped 5-fold OOF)")
    yr = df[REG_TARGET].to_numpy()
    Xr = df[NUM_REG + CAT]
    rfolds = list(GroupKFold(5).split(Xr, yr, groups))
    rows = []
    for name, (est, scale) in regressors().items():
        oof, t0 = np.zeros(len(yr)), time.time()
        for tr, va in rfolds:
            oof[va] = make_pipeline(prep(scale, NUM_REG), est).fit(Xr.iloc[tr], yr[tr]).predict(Xr.iloc[va])
        rows.append({"model": name, "MAE": mean_absolute_error(yr, oof), "RMSE": root_mean_squared_error(yr, oof),
                     "R2": r2_score(yr, oof), "fit+predict s": time.time() - t0})
    print(pd.DataFrame(rows).sort_values("MAE").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Always compare against the dummy row: a model that cannot beat 'predict the mean' has learned nothing.")

    banner("C · Reading coefficients (fitted on all rows, standardised numbers)")
    from sklearn.linear_model import LinearRegression, LogisticRegression
    pre_r = prep(True, NUM_REG).fit(Xr)
    lin = LinearRegression().fit(pre_r.transform(Xr), yr)
    names_r = [n.split("__", 1)[1] for n in pre_r.get_feature_names_out()]
    print(pd.DataFrame({"feature": names_r, "linear: medications per +1 SD": lin.coef_}).round(3).to_string(index=False))
    pre = prep(True).fit(X)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    log = LogisticRegression(max_iter=2000).fit(pre.transform(X), y)
    print()
    print(pd.DataFrame({"feature": names, "logistic: odds ratio per +1 SD": np.exp(log.coef_[0])}).round(3)
          .to_string(index=False))
    print("Linear: +1 standard deviation in the feature changes the predicted count by that many medications.\n"
          "Logistic: odds of readmission are multiplied by the odds ratio (1.0 = no effect). "
          "These are associations, not causes.")

    banner("D · Unsupervised: clustering, PCA, anomaly detection (no labels used)")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture
    Z = prep(True).fit_transform(X)
    for k in (2, 3, 4, 5, 6):
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
        print(f"KMeans k={k}: silhouette {silhouette_score(Z, lab):.3f}")
    gmm = GaussianMixture(n_components=3, random_state=SEED).fit(Z)
    print(f"Gaussian mixture (3 components): BIC {gmm.bic(Z):.0f} (lower is better when comparing component counts)")
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    print("PCA cumulative variance explained by first 1-5 components:", np.round(cum[:5], 3).tolist())
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=SEED).fit(Z)
    flagged = df[iso.predict(Z) == -1]
    print(f"Isolation forest flagged {len(flagged)} unusual admissions (2% contamination). Medians, flagged vs all:")
    print(pd.DataFrame({"flagged": flagged[NUM].median(), "all": df[NUM].median()}).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
