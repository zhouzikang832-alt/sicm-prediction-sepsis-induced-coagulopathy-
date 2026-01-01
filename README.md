# Sepsis-Associated Coagulopathy ICU Risk Prediction App

An interactive web-based clinical decision support tool for predicting **ICU admission risk** in patients with **sepsis-associated coagulopathy (SAC)**, incorporating **individual-level SHAP explainability**.

---

## 🔬 Project Overview

This application implements a machine learning model developed from large-scale ICU databases to estimate the probability of ICU admission among patients with sepsis-associated coagulopathy using **day-1 clinical variables**.

Key features include:
- Accurate ICU admission risk prediction
- Individualized, patient-level model interpretation using SHAP
- Web-based deployment for real-time clinical demonstration

---

## 🧠 Model Details

- **Model type**: XGBoost (tree-based ensemble)
- **Input features**: Day-1 laboratory and vital sign variables
- **Outcome**: ICU admission (binary)
- **Performance**:  
  - AUROC ≈ 0.94 (internal validation)
- **Explainability**:  
  - SHAP TreeExplainer  
  - Single-patient (local) explanation via waterfall plot

---

## 🖥️ Application Features

- Manual input of clinical variables
- Real-time prediction of ICU admission probability
- Risk stratification (Low / Moderate / High)
- Visualization of top contributing features for each individual patient
- SHAP waterfall plot for transparent clinical interpretation

---

## 📁 Repository Structure

