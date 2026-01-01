import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import ast
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
# 递归拆 Pipeline（SHAP 兜底）
# =========================
def unwrap_pipeline(obj):
    if isinstance(obj, Pipeline):
        return unwrap_pipeline(list(obj.named_steps.values())[-1])
    return obj

def find_preprocessor(obj):
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "transform"):
                return step
    return None

final_model = None
preprocessor = None

if isinstance(model_obj, Pipeline):
    preprocessor = find_preprocessor(model_obj)
    final_model = unwrap_pipeline(model_obj)

elif isinstance(model_obj, dict):
    for v in model_obj.values():
        if isinstance(v, Pipeline):
            preprocessor = find_preprocessor(v)
            final_model = unwrap_pipeline(v)
            break
        if hasattr(v, "predict_proba"):
            final_model = v

if final_model is None or isinstance(final_model, Pipeline):
    st.error("❌ Failed to extract final model for SHAP")
    st.stop()

# =========================
# 特征（固定 20 个）
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
# 🔑 超鲁棒数值解析函数（核心修复点）
# =========================
def robust_float(x):
    """
    将各种奇葩输入安全转为 float
    """
    if x is None:
        return np.nan

    # numpy array
    if isinstance(x, (np.ndarray, list)):
        if len(x) == 0:
            return np.nan
        return robust_float(x[0])

    # 字符串
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return np.nan

        # 形如 "[3.1092438E-1]"
        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                return robust_float(parsed)
            except Exception:
                return np.nan

        try:
            return float(x)
        except Exception:
            return np.nan

    # 普通数值
    try:
        return float(x)
    except Exception:
        return np.nan

# =========================
# 输入区域
# =========================
st.sidebar.header("📥 Patient Variables")

input_data = {}
for feat in feature_names:
    input_data[feat] = st.sidebar.text_input(feat, "")

X_input = pd.DataFrame([input_data])

# 强制逐元素清洗
for col in X_input.columns:
    X_input[col] = X_input[col].apply(robust_float)

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
        st.subheader("🧠 SHAP Explanation (Single Patient)")

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

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
