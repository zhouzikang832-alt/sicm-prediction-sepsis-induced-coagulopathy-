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
# 加载完整 Pipeline（不拆）
# =========================
@st.cache_resource
def load_pipeline():
    return joblib.load("best_model_XGBoost.pkl")

model = load_pipeline()

# =========================
# 特征（必须与训练一致）
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
# 输入区域（只允许 float）
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

X_input = pd.DataFrame([values], columns=feature_names, dtype=float)

# =========================
# 预测 + SHAP（通用 Explainer）
# =========================
if st.button("🔍 Predict & Explain"):
    try:
        # ---------- 预测 ----------
        prob = model.predict_proba(X_input)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric("Predicted Mortality Risk", f"{prob:.3f}")

        # ---------- SHAP ----------
        st.subheader("🧠 SHAP Explanation (Pipeline-compatible)")

        # 🔥 关键：用通用 Explainer
        explainer = shap.Explainer(
            model.predict_proba,
            X_input,
            algorithm="auto"
        )

        shap_values = explainer(X_input)

        # 取正类
        if shap_values.values.ndim == 3:
            shap_vals = shap_values.values[0, :, 1]
            base_val = shap_values.base_values[0, 1]
        else:
            shap_vals = shap_values.values[0]
            base_val = shap_values.base_values[0]

        # Waterfall
        fig1 = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=X_input.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig1)

        # Bar
        fig2 = plt.figure(figsize=(9, 5))
        shap.plots.bar(
            shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=X_input.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction or SHAP explanation failed: {e}")
