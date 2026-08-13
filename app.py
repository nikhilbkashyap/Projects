"""
KCET Cutoff Predictor - Streamlit App
========================================
WHAT THIS FILE DOES:
  The website itself. Loads the trained model (via predict_utils.py),
  lets a user pick a college/branch/category/round/year, and shows a
  predicted closing rank alongside the real historical trend for that
  exact combination - so the user can judge the prediction against real
  data, not just trust a number in isolation.

WHY THE UI LOGIC IS THIN:
  All the actual prediction logic lives in predict_utils.py, which was
  tested standalone (see its __main__ block and 04_application/README.md
  for the bug that testing caught). This file is deliberately just
  wiring: collect inputs, call predict_utils functions, display results.

HOW TO RUN:
  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from predict_utils import load_artifacts, predict_cutoff, get_historical_trend

st.set_page_config(page_title="KCET Cutoff Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, categories_map, feature_cols, lookup, metrics, historical = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("🎓 KCET Cutoff Predictor")
st.caption(
    "Predicts KCET closing ranks based on historical data. "
    "Built as a personal/academic project - not an official KEA tool."
)

is_sample_data = len(historical) > 0 and "SAMPLE_DATA" in str(historical["source_file"].iloc[0])

with st.expander("⚠️ Important: what this prediction is and isn't", expanded=False):
    st.markdown(
        f"""
- This is currently trained on **{'synthetic sample data' if is_sample_data else 'the data currently in the pipeline'}** -
  see `01_data_collection/README.md` for how to swap in real scraped KEA data.
- Model accuracy on held-out 2025 data: **R² = {metrics['r2']}**, average error
  (MAE) = **{metrics['mae']} ranks**. That means predictions can reasonably
  be off by several hundred ranks - use this as a *guide*, not a
  guarantee, especially for borderline cases.
- KEA's actual cutoffs depend on that year's applicant volume, seat
  changes, and policy shifts - factors this model does not have access to.
        """
    )

st.subheader("Get a prediction")

col1, col2 = st.columns(2)
with col1:
    college = st.selectbox("College", lookup["colleges"])
    branch = st.selectbox("Branch", lookup["branches"])
with col2:
    category = st.selectbox("Category", lookup["categories"])
    round_name = st.selectbox("Round", ["1", "2", "Final"])

year = st.slider("Predict for year", min_value=2025, max_value=2028, value=2026)

if st.button("Predict Cutoff Rank", type="primary"):
    prediction = predict_cutoff(model, categories_map, feature_cols, college, branch, category, round_name, year)

    st.metric(
        label=f"Predicted closing rank — {college}, {branch}, {category}, Round {round_name}, {year}",
        value=f"{prediction:,}",
    )

    trend = get_historical_trend(historical, college, branch, category)

    if len(trend) > 0:
        st.subheader("How this compares to real historical data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f"{r['year']} R{r['round']}" for _, r in trend.iterrows()],
            y=trend["closing_rank"],
            mode="lines+markers",
            name="Historical closing rank",
            line=dict(color="#1f77b4"),
        ))
        fig.add_trace(go.Scatter(
            x=[f"{year} R{round_name} (predicted)"],
            y=[prediction],
            mode="markers",
            name="Prediction",
            marker=dict(color="red", size=14, symbol="star"),
        ))
        fig.update_layout(
            yaxis_title="Closing Rank (lower = more competitive)",
            xaxis_title="Year / Round",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data found for this exact combination to compare against.")

st.divider()
with st.expander("Model performance details"):
    st.json(metrics)
    st.caption(
        "Evaluated by training on 2021-2024 data and testing ONLY on "
        "2025 data the model never saw during training - see "
        "03_model_development/README.md for why this split matters."
    )
