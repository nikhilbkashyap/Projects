"""
Model Development
===================
WHAT THIS FILE DOES:
  Loads the clean data from Part 2 (02_data_processing/output/
  kcet_cleaned.csv) and trains a model that predicts closing_rank given
  (year, round, college, branch, category). Saves the trained model plus
  everything Part 4 (the app) needs to use it.

WHY GRADIENT BOOSTING (XGBoost), NOT SOMETHING FANCIER:
  This is a small-data, tabular, mostly-categorical problem (~10 colleges
  x 6 branches x 9 categories x 5 years x 3 rounds = a few thousand
  distinct combinations, not millions of rows of continuous signal).
  Gradient-boosted trees are the standard, well-proven choice for this
  shape of problem - they handle categorical-heavy tabular data well
  without needing huge amounts of data, and they're fast to train and
  easy to explain (unlike a neural net, which would be overkill and
  likely to overfit on data this size anyway).

WHY THE TRAIN/TEST SPLIT IS BY YEAR, NOT RANDOM:
  A random 80/20 split would let the model "see" 2025 data for some
  colleges while being tested on 2025 data for others - that leaks
  future information and makes the test score look better than the
  model actually is. Since the real use case is "predict a cutoff you
  haven't seen yet," the fair test is: train on 2021-2024, test on 2025
  entirely held out. This gives an honest sense of how well the model
  would have done predicting last year blind.

WHAT GETS SAVED (all in output/, all needed by Part 4):
  - model.pkl               the trained XGBoost model
  - encoders.pkl            fixed category lists for college/branch/category/
                             round (needed to encode new prediction inputs
                             the exact same way the model was trained on)
  - metrics.json            honest performance numbers - MAE, RMSE, R2
  - feature_importance.csv  which inputs matter most for predictions
  - college_branch_lookup.json  valid dropdown options for the app

HOW TO RUN:
  python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

INPUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "02_data_processing", "output", "kcet_cleaned.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cleaned data not found at {INPUT_CSV}\n"
            "Run 02_data_processing/clean_data.py first."
        )
    return pd.read_csv(INPUT_CSV)


def build_features(df, categories_map=None, fit=True):
    """
    Converts text columns (college_name, branch, category, round) into
    pandas 'category' dtype columns that XGBoost can split on natively.

    IMPORTANT - WHY NOT LabelEncoder (the earlier, buggy approach):
    LabelEncoder assigns numbers alphabetically - e.g. category becomes
    GM=6, SC=7, ST=8, 1G=0. XGBoost then treats that as if 6 < 7 < 8 is
    a meaningful ORDER, when really these are just different unordered
    labels. That false ordering measurably corrupted predictions during
    testing (the same training row predicted 921 vs ~14 depending on
    unrelated tree splits). Using true categorical dtype instead lets
    XGBoost split cleanly on "is this GM or not", with no fake order
    assumption - this is what actually fixed it.

    If `fit=True`, the exact set of categories seen in this data becomes
    the fixed category list, stored in `categories_map`. If `fit=False`,
    the SAME category list (from training) is reused - any value not in
    that list becomes NaN, which XGBoost handles as "missing" and routes
    down whichever branch it learned was safest during training, rather
    than crashing or silently guessing.
    """
    df = df.copy()
    categorical_cols = ["college_name", "branch", "category", "round"]

    if fit:
        categories_map = {}
        for col in categorical_cols:
            cats = sorted(df[col].astype(str).unique().tolist())
            categories_map[col] = cats
            df[col + "_cat"] = pd.Categorical(df[col].astype(str), categories=cats)
    else:
        for col in categorical_cols:
            cats = categories_map[col]
            df[col + "_cat"] = pd.Categorical(df[col].astype(str), categories=cats)

    feature_cols = [
        "year",
        "college_name_cat",
        "branch_cat",
        "category_cat",
        "round_cat",
    ]
    X = df[feature_cols]
    return X, categories_map, feature_cols


def train_and_evaluate(df):
    # Time-based split: train on everything before 2025, test on 2025.
    train_df = df[df["year"] < 2025]
    test_df = df[df["year"] == 2025]

    print(f"Train rows: {len(train_df)} (years {sorted(train_df['year'].unique())})")
    print(f"Test rows:  {len(test_df)} (year 2025, fully held out)")

    X_train, categories_map, feature_cols = build_features(train_df, fit=True)
    y_train = train_df["closing_rank"]

    X_test, _, _ = build_features(test_df, categories_map=categories_map, fit=False)
    y_test = test_df["closing_rank"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        enable_categorical=True,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 1, None)  # a predicted rank can't be <= 0

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    metrics = {
        "mae": round(float(mae), 1),
        "rmse": round(float(rmse), 1),
        "r2": round(float(r2), 4),
        "mape_percent": round(float(mape), 1),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_year_held_out": 2025,
        "note": (
            "Trained on synthetic sample data (see 01_data_collection/"
            "generate_sample_data.py). These numbers describe the "
            "PIPELINE's behavior, not real-world prediction accuracy. "
            "Re-run this whole script after swapping in real scraped "
            "data to get metrics that mean something."
        ),
    }

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, categories_map, metrics, importance, feature_cols


def save_lookup_tables(df):
    """Saves the valid dropdown values for Part 4's app - college names,
    branches, categories - so the app's UI only ever offers combinations
    that actually exist in the data."""
    lookup = {
        "colleges": sorted(df["college_name"].unique().tolist()),
        "branches": sorted(df["branch"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()),
        "years_in_training_data": sorted(df["year"].unique().tolist()),
    }
    with open(os.path.join(OUTPUT_DIR, "college_branch_lookup.json"), "w") as f:
        json.dump(lookup, f, indent=2)
    return lookup


def main():
    print("Loading cleaned data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows.")

    print("\nTraining model (train: pre-2025, test: 2025 held out)...")
    model, categories_map, metrics, importance, feature_cols = train_and_evaluate(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))
    joblib.dump(categories_map, os.path.join(OUTPUT_DIR, "encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, "feature_cols.pkl"))
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    print("\nSaving full cleaned data lookup tables for the app...")
    save_lookup_tables(df)

    print("\n=== RESULTS (held-out 2025 data) ===")
    print(f"  MAE  (avg rank error):      {metrics['mae']}")
    print(f"  RMSE:                       {metrics['rmse']}")
    print(f"  R2:                         {metrics['r2']}")
    print(f"  MAPE:                       {metrics['mape_percent']}%")
    print("\nFeature importance:")
    print(importance.to_string(index=False))

    print(f"\nAll artifacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
