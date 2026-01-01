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
# 加载模型对象
# =========================
@st.cache_resource
def load_model():
    return joblib.load("best_model_XGBoost.pkl")

model_obj = load_model()

# =========================
# 🔑 递归拆 Pipeline（核心兜底）
# =========================
def unwrap_pipeline(obj):
    """
    不断拆 Pipeline，直到拿到最底层的模型
    """
    if isinstance(obj, Pipeline):
        # 取最后一个 step
        last_step = list(obj.named_steps.values())[-1]
        return unwrap_pipeline(last_step)
    else:
        return obj

def find_preprocessor(obj):
    """
    从 Pipeline 中找到第一个 transform（如 SimpleImputer）
    """
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "transform"):
                return step
    return None

# =========================
# 解析模型结构
# =========================
final_model = None
preprocessor = None

# 情况 1：直接是 Pipeline
if isinstance(model_obj, Pipeline):
    preprocessor = find_preprocessor(model_obj)
    final_model = unwrap_pipeline(model_obj)

# 情况 2：是 dict
elif isinstance(model_obj, dict):
    for v in model_obj.values():
        if isinstance(v, Pipeline):
            preprocessor = find_preprocessor(v)
            final_model = unwrap_pipeline(v)
            break

    # 如果还没找到，再兜底
    if final_model is None:
        for v in model_obj.values():
            if hasattr(v, "predict_proba"):
                final_model = unwrap_pipeline(v)
            elif hasattr(v, "transform"):
                preprocessor = v

# 最终兜底
if final_model is None or isinstance(final_model, Pipeline):
    st.error("❌ 未能解析出可用于 SHAP 的最终模型（非 Pipeline）")
    st.stop()

# =========================
# 特征名（固定 20 个）
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
