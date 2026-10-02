"""
🤖 15-Minute Automated Tabular Stacking Pipeline (AutoGluon)
============================================================
Uses AutoGluon Tabular to automatically train, tune, and ensemble
LightGBM, CatBoost, XGBoost, and neural networks using multi-layer stacking.

Requirements:
    pip install autogluon

Run:
    python automl-autogluon.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_or_generate_data():
    """Generates realistic tabular data or loads from train.csv if present."""
    if os.path.exists("train.csv"):
        print("📁 Loading local 'train.csv'...")
        return pd.read_csv("train.csv")
    
    print("⚡ Generating synthetic 5,000-row tabular dataset for benchmarking...")
    X, y = make_classification(
        n_samples=5000,
        n_features=12,
        n_informative=8,
        weights=[0.85, 0.15],
        random_state=42
    )
    cols = [
        "risk_score_a", "utilization_rate", "transaction_velocity",
        "tenure_months", "credit_limit_ratio", "past_delinquency_count",
        "avg_daily_balance", "customer_age", "monthly_inflow",
        "support_ticket_count", "signal_x", "signal_y"
    ]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    df["region"] = np.random.choice(["Urban", "Suburban", "Rural"], size=len(df))
    return df

def run_autogluon_training(df, target_col="target", time_limit_seconds=300):
    """
    Trains AutoGluon TabularPredictor with multi-layer stacking.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("\n⚠️ AutoGluon is not installed in this environment.")
        print("👉 Install it with: pip install autogluon")
        print("   No results are shown without AutoGluon: this script never prints made-up scores.")
        sys.exit(2)

    # Split train and validation holdout
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    print(f"📊 Training rows: {len(train_data):,} | Test rows: {len(test_data):,}")

    output_dir = os.path.join("artifacts", "autogluon_models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Launching AutoGluon (Time budget: {time_limit_seconds} seconds)...")
    predictor = TabularPredictor(
        label=target_col,
        eval_metric="roc_auc",
        path=output_dir
    ).fit(
        train_data=train_data,
        time_limit=time_limit_seconds,
        presets="best_quality" # Enables multi-layer stacking with out-of-fold bag ensembling!
    )

    # Leaderboard
    print("\n🏆 Final Model Leaderboard:")
    leaderboard = predictor.leaderboard(test_data, silent=False)
    
    # Save test predictions
    y_pred_probs = predictor.predict_proba(test_data)
    test_data["predicted_probability"] = y_pred_probs[1]
    test_data.to_csv("artifacts/autogluon_predictions.csv", index=False)
    print("\n✅ Predictions saved to 'artifacts/autogluon_predictions.csv'!")
    return predictor

def main():
    print("=" * 65)
    print("🤖 AutoGluon Multi-Layer Tabular Stacking Pipeline")
    print("=" * 65)
    df = load_or_generate_data()
    # Default to 300s (5 mins); adjust to 900s (15 mins) for final competition submission
    run_autogluon_training(df, target_col="target", time_limit_seconds=300)

if __name__ == "__main__":
    main()
