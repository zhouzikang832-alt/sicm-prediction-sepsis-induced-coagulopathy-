import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="SICM Mortality Prediction",
    layout="wide"
)

st.title("🫀 SICM Mortality Prediction with SHAP Explanation")

# =========================
# 加载模型 bundle（dict）
# =========================
@st.cache_resource
def load_bundle():
    bundle = joblib.load("best_model_XGBoost.pkl")
    return bundle

bundle = load_bundle()

# 从 dict 中取组件（关键修复）
model = bundle["model"]
preprocessor = bundle.get("preprocessor", None)

# =========================
# 加载特征名
# =========================
@st.cache_data
def load_feature_names():
    with open("feature_names.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

feature_names = load_feature_names()

# =========================
# 输入区域
# =========================
st.sidebar.header("📥 Patient Variables")

input_data = {}
for feat in feature_names:
    input_data[feat] = st.sidebar.text_input(feat, "")

# =========================
# 输入清洗函数（核心防炸）
# =========================
def safe_float(x):
    """
    将 '[3.1E-1]'、'0.3'、array([0.3]) 等
    统一转为 float，异常值返回 NaN
    """
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
        prob = model.predict_proba(X_processed)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric(
            label="Predicted Mortality Risk",
            value=f"{prob:.3f}"
        )

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Single-Patient Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        # 二分类模型取正类
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # ===== Waterfall Plot =====
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

        # ===== Bar Plot =====
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
