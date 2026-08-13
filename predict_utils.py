"""
Prediction Logic (UI-independent)
====================================
WHAT THIS FILE DOES:
  Loads the trained model + encoders from Part 3 and exposes one function,
  predict_cutoff(), that takes plain human inputs (college name, branch,
  category, round, year) and returns a predicted closing rank.

WHY THIS IS SEPARATE FROM app.py:
  app.py (the Streamlit UI) imports this file rather than containing the
  prediction logic directly. That split means this logic can be tested
  with plain Python - no browser, no Streamlit server needed - which is
  exactly how it's verified below in the __main__ block. Mixing UI code
  and logic in one file makes the logic much harder to test in isolation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "03_model_development", "output"
)
HISTORICAL_CSV = os.path.join(
    os.path.dirname(__file__), "..", "02_data_processing", "output", "kcet_cleaned.csv"
)


def load_artifacts():
    """Loads everything Part 3 produced. Raises a clear error if Part 3
    hasn't been run yet, rather than a confusing pickle/file-not-found
    trace."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            "Run 03_model_development/train_model.py first."
        )
    model = joblib.load(model_path)
    categories_map = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    with open(os.path.join(MODEL_DIR, "college_branch_lookup.json")) as f:
        lookup = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    historical = pd.read_csv(HISTORICAL_CSV)
    return model, categories_map, feature_cols, lookup, metrics, historical


def predict_cutoff(model, categories_map, feature_cols, college, branch, category, round_name, year):
    """
    Predicts closing_rank for one specific combination.

    Uses the same pandas-categorical encoding as training (see
    train_model.py's build_features docstring for why plain numeric
    label encoding was wrong here). A value not seen during training
    (e.g. a college that didn't exist in the historical data) becomes
    NaN, which XGBoost treats as "missing" and routes the same way it
    learned to during training - safe, no crash, no fake category.
    """
    row = pd.DataFrame(index=[0])
    row["year"] = year
    for col, key, value in [
        ("college_name_cat", "college_name", college),
        ("branch_cat", "branch", branch),
        ("category_cat", "category", category),
        ("round_cat", "round", round_name),
    ]:
        cats = categories_map[key]
        value = str(value)
        row[col] = pd.Categorical([value if value in cats else None], categories=cats)
    row = row[feature_cols]

    pred = model.predict(row)[0]
    return max(1, round(float(pred)))


def get_historical_trend(historical_df, college, branch, category):
    """Returns the real historical closing ranks for one combination,
    across all years/rounds - used to plot the trend alongside the
    prediction so the user can see how the model's guess compares to
    the actual pattern it learned from."""
    subset = historical_df[
        (historical_df["college_name"] == college)
        & (historical_df["branch"] == branch)
        & (historical_df["category"] == category)
    ].copy()
    return subset.sort_values(["year", "round_sort_order"])


if __name__ == "__main__":
    # Smoke test: run this file directly (no Streamlit needed) to verify
    # the whole prediction path works before ever touching the UI.
    print("Loading artifacts...")
    model, categories_map, feature_cols, lookup, metrics, historical = load_artifacts()
    print(f"  Model loaded. Trained on years: {lookup['years_in_training_data']}")

    test_college = lookup["colleges"][0]
    test_branch = "Computer Science" if "Computer Science" in lookup["branches"] else lookup["branches"][0]
    test_category = "GM" if "GM" in lookup["categories"] else lookup["categories"][0]

    print(f"\nTest prediction: {test_college} | {test_branch} | {test_category} | Round 1 | 2026")
    pred = predict_cutoff(model, categories_map, feature_cols, test_college, test_branch, test_category, "1", 2026)
    print(f"  Predicted closing rank: {pred}")

    print("\nTest with an UNSEEN college (should not crash):")
    pred_unseen = predict_cutoff(model, categories_map, feature_cols, "Some College Not In Training Data", test_branch, test_category, "1", 2026)
    print(f"  Predicted closing rank (fallback path): {pred_unseen}")

    print("\nHistorical trend for the same combination:")
    trend = get_historical_trend(historical, test_college, test_branch, test_category)
    print(trend[["year", "round", "closing_rank"]].to_string(index=False))

    print("\nAll checks passed.")
