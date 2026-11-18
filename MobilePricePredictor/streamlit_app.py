"""Streamlit front end for the Mobile Price Predictor."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobile_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "mobile_phones.csv"
TEMPLATE_PATH = BASE_DIR / "static" / "phone_template.csv"

CATEGORICAL = ["brand", "model"]
NUMERIC = ["rating", "ram_gb", "storage_gb", "screen_inch",
           "main_camera_mp", "front_camera_mp", "battery_mah"]
FEATURES = CATEGORICAL + NUMERIC
LABELS = {
    "rating": "User rating (1-5)",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Storage (GB)",
    "screen_inch": "Screen size (inches)",
    "main_camera_mp": "Rear camera (MP)",
    "front_camera_mp": "Selfie camera (MP)",
    "battery_mah": "Battery (mAh)",
}
RANGES = {
    "rating": (1.0, 5.0),
    "ram_gb": (1.0, 32.0),
    "storage_gb": (1.0, 2048.0),
    "screen_inch": (3.0, 10.0),
    "main_camera_mp": (1.0, 300.0),
    "front_camera_mp": (1.0, 100.0),
    "battery_mah": (500.0, 20000.0),
}


def price_category(price):
    if price < 10000:
        return "Budget"
    if price < 25000:
        return "Mid-range"
    if price < 50000:
        return "Premium"
    return "Flagship"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict(df, bundle):
    """Return predicted prices in INR (never negative)."""
    return np.maximum(0, bundle["model"].predict(df[FEATURES]))


st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")
st.title("📱 Mobile Price Predictor")

bundle = load_bundle()
data = load_data()
metrics = load_metrics()
uncertainty = float(bundle["uncertainty_abs"])

tab_single, tab_batch, tab_model = st.tabs(
    ["Single phone", "Batch CSV upload", "Model details"]
)

# ---------- Tab 1: one phone ----------
with tab_single:
    st.write("Pick a brand and model, adjust the specs, then click **Predict price**.")
    brands = sorted(data["brand"].dropna().unique())
    brand = st.selectbox("Brand", brands)
    models = sorted(data.loc[data["brand"] == brand, "model"].dropna().unique())
    model_name = st.selectbox("Mobile model", models)

    # Start the spec fields at the median value for this brand's data
    subset = data[(data["brand"] == brand) & (data["model"] == model_name)]
    source = subset if not subset.empty else data
    defaults = {c: float(source[c].median()) for c in NUMERIC}

    with st.form("single_form"):
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(NUMERIC):
            lo, hi = RANGES[col]
            with cols[i % 3]:
                values[col] = st.number_input(
                    LABELS[col],
                    min_value=lo,
                    max_value=hi,
                    value=min(max(defaults[col], lo), hi),
                )
        submitted = st.form_submit_button("Predict price")

    if submitted:
        row = pd.DataFrame([{"brand": brand, "model": model_name, **values}])
        price = float(predict(row, bundle)[0])
        low = max(0.0, price - uncertainty)
        high = price + uncertainty
        c1, c2 = st.columns(2)
        c1.metric("Estimated price", f"₹{price:,.0f}")
        c2.metric("Category", price_category(price))
        st.info(
            f"Likely range: ₹{low:,.0f} to ₹{high:,.0f} "
            "(based on 90% of test-set errors)."
        )

# ---------- Tab 2: batch upload ----------
with tab_batch:
    st.write("Upload a CSV with the same columns as the template to predict many phones at once.")
    if TEMPLATE_PATH.exists():
        st.download_button(
            "Download CSV template",
            TEMPLATE_PATH.read_bytes(),
            file_name="phone_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                batch["predicted_price_inr"] = predict(batch, bundle).round(0)
                st.dataframe(batch, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    batch.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:  # show a readable message instead of a crash
            st.error(f"Could not process this file: {exc}")

# ---------- Tab 3: model details ----------
with tab_model:
    st.write(f"Best model: **{metrics['models'][metrics['best_model']]['label']}**")
    st.write(f"Training rows: {metrics['rows']}")
    table = pd.DataFrame(metrics["models"]).T[
        ["label", "cv_r2_mean", "test_r2", "test_mae_inr", "test_mape_pct", "within_10pct"]
    ]
    st.dataframe(table, use_container_width=True)
    st.caption(
        "Limitation: the data has no dated price history, so this tool estimates "
        "today's price from specs. It does not forecast future prices."
    )
