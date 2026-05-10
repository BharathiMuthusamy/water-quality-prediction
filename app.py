import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")

# ------------------------------------------------
# ✅ Load Model + Scaler with Error Handling
# ------------------------------------------------
try:
    if not os.path.exists("models/best_model.pkl"):
        raise FileNotFoundError("Model file not found. Please run train.py first.")
    if not os.path.exists("models/scaler.pkl"):
        raise FileNotFoundError("Scaler file not found. Please run train.py first.")
    if not os.path.exists("models/feature_names.pkl"):
        raise FileNotFoundError("Feature names file not found. Please run train.py first.")
    
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    
    st.title("💧 Water Potability Predictor")
    st.success("✅ Model Loaded Successfully")
except FileNotFoundError as e:
    st.title("💧 Water Potability Predictor")
    st.error(f"❌ Error: {e}")
    st.stop()
except Exception as e:
    st.title("💧 Water Potability Predictor")
    st.error(f"❌ Error loading model files: {str(e)}")
    st.stop()

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

    # ✅ WHO-Based Safety Warnings
    st.subheader("📋 Parameter Analysis")
    
    warnings = []
    safe_params = []
    
    # Check each parameter against WHO safe ranges
    if ph < 6.5 or ph > 8.5:
        warnings.append(f"⚠️ **pH {ph:.2f}** - Outside safe range (6.5 – 8.5)")
    else:
        safe_params.append(f"✅ pH {ph:.2f} - Safe")
    
    if Hardness < 60 or Hardness > 180:
        warnings.append(f"⚠️ **Hardness {Hardness:.2f}** - Outside safe range (60 – 180 mg/L)")
    else:
        safe_params.append(f"✅ Hardness {Hardness:.2f} - Safe")
    
    if Solids > 1000:
        warnings.append(f"⚠️ **Solids {Solids:.2f}** - Exceeds safe limit (<1000 mg/L)")
    else:
        safe_params.append(f"✅ Solids {Solids:.2f} - Safe")
    
    if Chloramines < 2 or Chloramines > 4:
        warnings.append(f"⚠️ **Chloramines {Chloramines:.2f}** - Outside safe range (2 – 4 mg/L)")
    else:
        safe_params.append(f"✅ Chloramines {Chloramines:.2f} - Safe")
    
    if Sulfate > 250:
        warnings.append(f"⚠️ **Sulfate {Sulfate:.2f}** - Exceeds safe limit (<250 mg/L)")
    else:
        safe_params.append(f"✅ Sulfate {Sulfate:.2f} - Safe")
    
    if Conductivity > 1500:
        warnings.append(f"⚠️ **Conductivity {Conductivity:.2f}** - Exceeds safe limit (<1500 µS/cm)")
    else:
        safe_params.append(f"✅ Conductivity {Conductivity:.2f} - Safe")
    
    if Organic_carbon > 5:
        warnings.append(f"⚠️ **Organic Carbon {Organic_carbon:.2f}** - Exceeds safe limit (<5 mg/L)")
    else:
        safe_params.append(f"✅ Organic Carbon {Organic_carbon:.2f} - Safe")
    
    if Trihalomethanes > 80:
        warnings.append(f"⚠️ **Trihalomethanes {Trihalomethanes:.2f}** - Exceeds safe limit (<80 µg/L)")
    else:
        safe_params.append(f"✅ Trihalomethanes {Trihalomethanes:.2f} - Safe")
    
    if Turbidity > 5:
        warnings.append(f"⚠️ **Turbidity {Turbidity:.2f}** - Exceeds safe limit (<5 NTU)")
    else:
        safe_params.append(f"✅ Turbidity {Turbidity:.2f} - Safe")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        if warnings:
            st.warning("**⚠️ Parameters Outside Safe Range:**\n" + "\n".join(warnings))
    
    with col2:
        if safe_params:
            st.success("**✅ Safe Parameters:**\n" + "\n".join(safe_params))
