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
# 加载模型
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model_XGBoost.pkl")

model_obj = load_model()

# =========================
# 解析模型结构
# =========================
# 情况 1：直接是 Pipeline（最常见）
if isinstance(model_obj, Pipeline):
    pipeline = model_obj
    final_model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["imputer"]

# 情况 2：是 dict
elif isinstance(model_obj, dict):

    pipeline = None
    final_model = None
    preprocessor = None

    # 先找 Pipeline
    for v in model_obj.values():
        if isinstance(v, Pipeline):
            pipeline = v
            final_model = pipeline.named_steps.get("model")
            preprocessor = pipeline.named_steps.get("imputer")
            break

    # 否则找模型和预处理
    if final_model is None:
        for v in model_obj.values():
            if hasattr(v, "predict_proba"):
                final_model = v
            elif hasattr(v, "transform"):
                preprocessor = v

# 兜底
if final_model is None:
    st.error("❌ 无法识别模型结构，请检查 best_model_XGBoost.pkl")
    st.stop()

# =========================
# 特征名（20 个，顺序必须一致）
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
# 输入区域
# =========================
st.sidebar.header("📥 Patient Variables")

input_data = {}
for feat in feature_names:
    input_data[feat] = st.sidebar.text_input(feat, "")

# =========================
# 输入清洗
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
        # ---------- 预处理 ----------
        if preprocessor is not None:
            X_processed = preprocessor.transform(X_input)
        else:
            X_processed = X_input.values

        # ---------- 预测 ----------
        prob = final_model.predict_proba(X_processed)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric("Predicted Mortality Risk", f"{prob:.3f}")

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Single-Patient Explanation")

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Waterfall
        fig1 = plt.figure(figsize=(9, 5))
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
        fig2 = plt.figure(figsize=(9, 5))
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
