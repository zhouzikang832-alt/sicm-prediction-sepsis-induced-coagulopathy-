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

obj = load_model()

# =========================
# 解析模型
# =========================
pipeline = None
final_model = None
preprocessor = None

if isinstance(obj, Pipeline):
    pipeline = obj
elif isinstance(obj, dict):
    for v in obj.values():
        if isinstance(v, Pipeline):
            pipeline = v
            break

if pipeline is None:
    st.error("❌ 未找到 sklearn Pipeline")
    st.stop()

# 强制约定命名
preprocessor = pipeline.named_steps.get("imputer")
final_model = pipeline.named_steps.get("model")

if final_model is None:
    st.error("❌ Pipeline 中未找到 model")
    st.stop()

# =========================
# 特征（顺序 = 训练顺序）
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
# 输入区域（只允许数值）
# =========================
st.sidebar.header("📥 Patient Variables")

input_values = []

for feat in feature_names:
    val = st.sidebar.number_input(
        label=feat,
        value=np.nan,
        step=0.01,
        format="%.5f"
    )
    input_values.append(val)

# DataFrame：从诞生起就是 float
X_input = pd.DataFrame([input_values], columns=feature_names, dtype=float)

# =========================
# 预测 + SHAP
# =========================
if st.button("🔍 Predict & Explain"):
    try:
        # ---------- 预处理 ----------
        X_processed = preprocessor.transform(X_input)

        # sklearn / xgboost 双保险
        X_processed = np.asarray(X_processed, dtype=float)

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
