import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =====================
# 页面设置
# =====================
st.set_page_config(
    page_title="Sepsis-Associated Coagulopathy ICU Risk Predictor",
    layout="centered"
)

st.title("🩸 Sepsis-Associated Coagulopathy")
st.subheader("ICU Admission Risk Prediction with SHAP Explanation")

st.markdown(
    """
    **Model overview**
    - Population: Sepsis-associated coagulopathy
    - Input: Day-1 laboratory & vital signs
    - Model: XGBoost (tree-based)
    - Performance: AUC = 0.942
    - Output: ICU admission probability + individualized explanation
    """
)

# =====================
# 加载模型（缓存）
# =====================
@st.cache_resource
def load_model():
    model_bundle = joblib.load("best_model_XGBoost.pkl")
    return model_bundle

model_bundle = load_model()
pipeline = model_bundle["pipeline"]
FEATURES = model_bundle["features"]

# 拆出 pipeline 内部组件
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["clf"]

# =====================
# 输入区
# =====================
st.markdown("## 🔬 Enter Day-1 Clinical Variables")

input_data = {}
col1, col2 = st.columns(2)

for i, feat in enumerate(FEATURES):
    with col1 if i % 2 == 0 else col2:
        input_data[feat] = st.number_input(
            label=feat,
            value=0.0,
            step=0.1,
            format="%.3f"
        )

X_input = pd.DataFrame([input_data])

# =====================
# 预测 + SHAP
# =====================
st.markdown("---")
if st.button("🚑 Predict ICU Risk & Explain", use_container_width=True):

    try:
        # ---------- 预测 ----------
        prob = pipeline.predict_proba(X_input)[0, 1]

        st.markdown("## 📊 Prediction Result")
        st.metric("Predicted ICU Admission Risk", f"{prob:.3f}")

        if prob < 0.20:
            st.success("🟢 Low risk")
        elif prob < 0.50:
            st.warning("🟡 Moderate risk")
        else:
            st.error("🔴 High risk")

        # ---------- SHAP 单病例解释 ----------
        st.markdown("## 🔍 Individualized SHAP Explanation")

        # 与训练阶段完全一致的预处理
        X_imp = imputer.transform(X_input)
        X_scaled = scaler.transform(X_imp)

        # TreeExplainer（适合 XGBoost / LightGBM / RF）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # 二分类：取 positive class
        if isinstance(shap_values, list):
            shap_vals_use = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_vals_use = shap_values
            expected_value = explainer.expected_value

        shap_df = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP value": shap_vals_use[0]
        })
        shap_df["|SHAP|"] = shap_df["SHAP value"].abs()
        shap_df = shap_df.sort_values("|SHAP|", ascending=False)

        # ---------- 表格形式（审稿人很爱） ----------
        st.markdown("### 🔝 Top contributing features")
        st.dataframe(
            shap_df.head(10)[["Feature", "SHAP value"]],
            use_container_width=True
        )

        # ---------- Waterfall Plot（单病例金标准） ----------
        st.markdown("### 🧠 SHAP Waterfall Plot")

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals_use[0],
                base_values=expected_value,
                data=X_input.iloc[0],
                feature_names=FEATURES
            ),
            max_display=10,
            show=False
        )
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            """
            **Interpretation**
            - Red features ↑ increase ICU risk  
            - Blue features ↓ decrease ICU risk  
            - Contributions are relative to the model baseline risk
            """
        )

    except Exception as e:
        st.error(f"Prediction or SHAP explanation failed: {e}")
