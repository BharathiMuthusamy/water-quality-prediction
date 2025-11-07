# 💧 Water Potability Predictor  
A Machine Learning web application that predicts whether water is **Safe** or **Not Safe** for drinking based on chemical properties.  
The model is trained using **CatBoost**, with preprocessing and balancing techniques, and the UI is built using **Streamlit**.

---

## 🚀 Overview

This project helps users check drinking water quality using 9 input parameters.  
The ML backend returns:

✅ Prediction (Safe / Not Safe)  
✅ Probability score  
✅ WHO-based warnings  
✅ Clean & fast Streamlit UI  

This version **does NOT use SHAP** — lightweight, stable, and deployable.

---

## 📂 Dataset

The dataset contains the following features:

- pH  
- Hardness  
- Solids  
- Chloramines  
- Sulfate  
- Conductivity  zz
- Organic Carbon  
- Trihalomethanes  
- Turbidity  
- Potability (target)

Missing values are filled using **median imputation**.

---

## 🧠 ML Pipeline

### ✅ 1. Data Cleaning
- Remove null values using median  
- Create engineered features (optional)  
- Save cleaned dataset

### ✅ 2. Preprocessing
- Train-test split  
- StandardScaler on numerical features  
- SMOTE-Tomek for class balancing

### ✅ 3. Model Training
Using **CatBoostClassifier**:

- 600 iterations  
- depth = 8  
- learning rate = 0.05  
- loss: CrossEntropy  

Artifacts saved:

- `models/best_catboost_model.pkl`  
- `models/scaler.pkl`  
- `models/feature_names.pkl`  

---

## 📁 Project Structure

📦 water-potability-predictor
├─ data/
│ └─ water_potability.csv
├─ processed_data/
│ └─ water_potability_cleaned.csv
├─ models/
│ ├─ best_catboost_model.pkl
│ ├─ scaler.pkl
│ └─ feature_names.pkl
├─ app.py
├─ train_model.py
└─ README.md

yaml
Copy code

---

## 🖥️ How to Run

### ✅ Install Dependencies
pip install -r requirements.txt

shell
Copy code

### ✅ Launch Streamlit App
streamlit run app.py

yaml
Copy code

---

## 🌡️ Prediction Logic

Your input values → Scaled → Sent to CatBoost model →  
Result:

✅ **Safe to Drink** (prob ≥ 0.50)  
❌ **Not Safe** (prob < 0.50)

The app also warns conditions like:

⚠️ pH not between **6.5 – 8.5**  
⚠️ Turbidity > **5 NTU**  
⚠️ Solids > **1500 mg/L**  

---

## ✅ Recommended Safe Ranges (Based on WHO)

| Feature | Ideal Range |
|--------|-------------|
| **pH** | 6.5 – 8.5 |
| **Hardness** | 60 – 180 mg/L |
| **Solids** | < 1500 mg/L |
| **Chloramines** | 2 – 4 mg/L |
| **Sulfate** | < 250 mg/L |
| **Organic Carbon** | < 5 mg/L |
| **Trihalomethanes** | < 80 µg/L |
| **Turbidity** | < 5 NTU |

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- CatBoost  
- Scikit-learn  
- Imbalanced-learn  
- NumPy & Pandas  

---

## 🌱 Future Enhancements

- 🔹 API endpoint  
- 🔹 IoT sensor live input  
- 🔹 Mobile version  
- 🔹 Auto-generate test report PDF  
- 🔹 Dashboard with charts  

---

## 👩‍💻 Author

**Bharathi M**  
AI & Data Science  
