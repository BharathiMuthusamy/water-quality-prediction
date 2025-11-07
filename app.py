import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")

# ------------------------------------------------
# ✅ Load Model + Scaler
# ------------------------------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("💧 Water Potability Predictor")
st.success("✅ Model Loaded Successfully")

# ------------------------------------------------
# ✅ UI Input Section
# ------------------------------------------------
st.header("Enter Water Quality Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0)
    Hardness = st.number_input("Hardness", 0.0, 1000.0, 150.0)
    Solids = st.number_input("Solids", 0.0, 60000.0, 5000.0)

with col2:
    Chloramines = st.number_input("Chloramines", 0.0, 15.0, 7.0)
    Sulfate = st.number_input("Sulfate", 0.0, 800.0, 300.0)
    Conductivity = st.number_input("Conductivity", 0.0, 2000.0, 450.0)

with col3:
    Organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 10.0)
    Trihalomethanes = st.number_input("Trihalomethanes", 0.0, 200.0, 70.0)
    Turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

# ------------------------------------------------
# ✅ Build Input Vector (correct order)
# ------------------------------------------------
input_data = np.array([[
    ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
    Organic_carbon, Trihalomethanes, Turbidity,
    abs(ph - 7),  # ph_diff_from_7
    Chloramines * Trihalomethanes,  # toxicity_mul
    1 if Turbidity > 4 else 0       # high_turbidity_flag
]])

# ------------------------------------------------
# ✅ Prediction
# ------------------------------------------------
if st.button("Predict"):
    scaled = scaler.transform(input_data)
    proba = float(model.predict_proba(scaled)[0][1])
    label = "✅ Safe to Drink" if proba >= 0.5 else "❌ Not Safe"

    st.subheader("Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Probability:** {proba:.3f}")

    # ✅ Extra Warnings
    warnings = []
    if ph < 6.5 or ph > 8.5:
        warnings.append("⚠️ pH is outside safe limit.")
    if Turbidity > 5:
        warnings.append("⚠️ High turbidity.")
    if Solids > 15000:
        warnings.append("⚠️ Very high dissolved solids.")

    if warnings:
        st.warning("\n".join(warnings))
