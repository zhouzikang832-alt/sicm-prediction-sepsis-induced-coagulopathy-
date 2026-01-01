import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="SICM Mortality Prediction",
    layout="wide"
)

st.title("🫀 SICM Mortality Prediction with SHAP Explanation")

# =========================
# 加载模型（只取最终模型）
# =========================
@st.cache_resource
def load_final_model():
    obj = joblib.load("best_model_XGBoost.pkl")

    # 如果是 Pipeline，取最后一步
    if isinstance(obj, Pipeline):
        return obj.steps[-1][1]

    # 如果是 dict，找能 predict_proba 的
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "predict_proba"):
                return v

    # 兜底
    if hasattr(obj, "predict_proba"):
        return obj

    raise RuntimeError("❌ 无法从 pkl 中提取最终模型")

model = load_final_model()

# =========================
# 特征（严格顺序）
# =========================
feature_names = [
    "RR",
    "DBP",
    "Absolute value of lymphocytes",
    "DD",
    "SPO2",
    "CKMB",
    "CRE",
    "SBP",
    "ALT",
    "LDH",
    "CRP",
    "Quantitative Myoglobin Assay",
    "HR",
    "PO2",
    "Absolute value of neutrophils",
    "IL-6",
    "AST",
    "PT",
    "INR1",
    "UREA"
]

# =========================
# 输入（只允许 float）
# =========================
st.sidebar.header("📥 Patient Variables")

values = []
for feat in feature_names:
    v = st.sidebar.number_input(
        feat,
        value=np.nan,
        step=0.01,
        format="%.6f"
    )
    values.append(v)

# numpy float32（XGBoost 原生）
X = np.array(values, dtype=np.float32).reshape(1, -1)

# =========================
# 预测 + SHAP
# =========================
if st.button("🔍 Predict & Explain"):
    try:
        # ---------- 预测 ----------
        prob = model.predict_proba(X)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric("Predicted Mortality Risk", f"{prob:.3f}")

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Explanation (Single Patient)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Waterfall
        fig1 = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig1)

        # Bar
        fig2 = plt.figure(figsize=(9, 5))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction or SHAP explanation failed: {e}")
