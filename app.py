import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="SICM Mortality Prediction with SHAP",
    layout="wide"
)

st.title("🫀 Sepsis-Induced Cardiomyopathy Mortality Risk Prediction")
st.markdown("Single-patient prediction with SHAP explanation")

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model_XGBoost.pkl")

model_pipeline = load_model()

# =========================
# 获取特征名
# =========================
if hasattr(model_pipeline, "feature_names_in_"):
    feature_names = model_pipeline.feature_names_in_
else:
    # 兜底（不推荐，但防炸）
    feature_names = model_pipeline.named_steps["model"].feature_name_

# =========================
# 输入区
# =========================
st.sidebar.header("📥 Patient Input")

input_data = {}

for feat in feature_names:
    input_data[feat] = st.sidebar.text_input(
        label=feat,
        value=""
    )

# =========================
# 输入清洗（关键修复点）
# =========================
def safe_float(x):
    """
    把 '[3.1E-1]' / '0.3' / array([0.3]) 全部兜成 float
    """
    if isinstance(x, str):
        x = x.strip().replace("[", "").replace("]", "")
    try:
        return float(x)
    except Exception:
        return np.nan

# 构造 DataFrame
X_input = pd.DataFrame([input_data])
X_input = X_input.applymap(safe_float)

# =========================
# 预测 & SHAP
# =========================
if st.button("🔍 Predict & Explain"):

    try:
        # -------- 预测 --------
        prob = model_pipeline.predict_proba(X_input)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric(
            label="Predicted Mortality Risk",
            value=f"{prob:.3f}"
        )

        # -------- SHAP 解释 --------
        st.subheader("🧠 SHAP Explanation (Single Patient)")

        # 取模型和预处理
        preprocessor = model_pipeline.named_steps.get("preprocessor", None)
        model = model_pipeline.named_steps["model"]

        if preprocessor is not None:
            X_processed = preprocessor.transform(X_input)
        else:
            X_processed = X_input.values

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        # 处理二分类情况
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # ===== Waterfall =====
        st.markdown("### 🔹 SHAP Waterfall Plot")

        fig1, ax1 = plt.subplots(figsize=(8, 5))
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

        # ===== Bar Plot =====
        st.markdown("### 🔹 SHAP Feature Importance (Single Case)")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
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
