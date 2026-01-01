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
# 加载 bundle
# =========================
@st.cache_resource
def load_bundle():
    return joblib.load("best_model_XGBoost.pkl")

bundle = load_bundle()

# =========================
# 🔑 自动解析 bundle 结构（核心）
# =========================
model = None
preprocessor = None

# 情况 1：bundle 本身就是 Pipeline
if isinstance(bundle, Pipeline):
    model = bundle
    preprocessor = None

# 情况 2：bundle 是 dict
elif isinstance(bundle, dict):

    # 优先找 Pipeline
    for v in bundle.values():
        if isinstance(v, Pipeline):
            model = v
            break

    # 否则找有 predict_proba 的对象
    if model is None:
        for v in bundle.values():
            if hasattr(v, "predict_proba"):
                model = v
            elif hasattr(v, "transform"):
                preprocessor = v

# 最终兜底
if model is None:
    st.error("❌ 未能从模型文件中识别可用于预测的模型对象")
    st.stop()

# =========================
# 特征名
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
        prob = model.predict_proba(X_processed)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric("Predicted Mortality Risk", f"{prob:.3f}")

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Single-Patient Explanation")

        explainer = shap.TreeExplainer(model)
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
