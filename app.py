import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="SICM Mortality Prediction",
    layout="wide"
)

st.title("🫀 SICM Mortality Prediction with SHAP")

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model_XGBoost.pkl")

model_pipeline = load_model()

# =========================
# 加载特征名（关键修复点）
# =========================
@st.cache_data
def load_feature_names():
    with open("feature_names.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

feature_names = load_feature_names()

# =========================
# 输入区
# =========================
st.sidebar.header("📥 Patient Variables")

input_data = {}
for feat in feature_names:
    input_data[feat] = st.sidebar.text_input(feat, "")

# =========================
# 输入清洗（防 '[3.1E-1]'）
# =========================
def safe_float(x):
    if isinstance(x, str):
        x = x.strip().replace("[", "").replace("]", "")
    try:
        return float(x)
    except Exception:
        return np.nan

X_input = pd.DataFrame([input_data])
X_input = X_input.applymap(safe_float)

# =========================
# 预测 + SHAP
# =========================
if st.button("🔍 Predict & Explain"):

    try:
        # ---------- 预测 ----------
        prob = model_pipeline.predict_proba(X_input)[0, 1]

        st.subheader("📊 Prediction")
        st.metric("Mortality Risk", f"{prob:.3f}")

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Explanation")

        preprocessor = model_pipeline.named_steps.get("preprocessor", None)
        model = model_pipeline.named_steps["model"]

        if preprocessor is not None:
            X_processed = preprocessor.transform(X_input)
        else:
            X_processed = X_input.values

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Waterfall
        fig1 = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_processed[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig1)

        # Bar
        fig2 = plt.figure(figsize=(8, 5))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_processed[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction or SHAP explanation failed: {e}")
