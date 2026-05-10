# 💧 Water Potability Predictor

A Machine Learning web application that predicts whether water is **Safe** or **Not Safe** for drinking based on chemical properties.
The model is trained using **CatBoost**, with preprocessing and balancing techniques, and deployed via **Streamlit** for an intuitive user interface.

---

## 🚀 Overview

This project evaluates drinking water quality using 9 chemical parameters and employs feature engineering to improve model performance. The ML backend provides:

✅ **Prediction Result** (Safe / Not Safe)  
✅ **Probability Score** (confidence level)  
✅ **WHO-based Safety Warnings**  
✅ **Fast & Clean Streamlit Interface**

---

## 📊 Model Performance

The trained CatBoost model achieves:

- **Accuracy:** ~70%  
- **F1-Score:** ~0.68  

*(These metrics will be updated after model retraining on production data)*

---

## 📂 Dataset

The dataset contains the following **9 water quality features**:

- **pH** — Acidity/Alkalinity  
- **Hardness** — Mineral content (mg/L)
- **Solids** — Total dissolved solids (mg/L)
- **Chloramines** — Disinfectant concentration (mg/L)
- **Sulfate** — Sulfate concentration (mg/L)
- **Conductivity** — Electrical conductivity (µS/cm)
- **Organic Carbon** — Organic matter (mg/L)
- **Trihalomethanes** — Disinfection byproducts (µg/L)
- **Turbidity** — Water cloudiness (NTU)
- **Potability** — Target variable (Safe=1, Not Safe=0)

**Data Preprocessing:** Missing values are imputed using median values to preserve data distribution.

---

## 🧠 ML Pipeline

### ✅ 1. Data Cleaning
- Median imputation for missing values
- Feature engineering (3 engineered features)
- Processed data saved for reproducibility

### ✅ 2. Feature Engineering
Three new features are created to improve model interpretability:
- **ph_diff_from_7** — Distance from neutral pH
- **toxicity_mul** — Interaction between Chloramines and Trihalomethanes
- **high_turbidity_flag** — Binary flag for high turbidity (>4 NTU)

### ✅ 3. Data Preprocessing
- Train-test split (80-20, stratified)
- StandardScaler normalization
- SMOTETomek for class balancing

### ✅ 4. Model Training
Using **CatBoostClassifier**:

- **Iterations:** 500  
- **Depth:** 7  
- **Learning Rate:** 0.05  
- **Loss Function:** CrossEntropy  
- **Random Seed:** 42 (for reproducibility)

**Saved Artifacts:**

- `models/best_model.pkl` — Trained CatBoost model
- `models/scaler.pkl` — StandardScaler for feature normalization
- `models/feature_names.pkl` — Feature column names  

---

## 📁 Project Structure

```
water-quality-prediction/
├─ data/
│  └─ water_potability.csv           # Raw dataset
├─ processed_data/
│  └─ water_potability_cleaned.csv   # Cleaned & processed data
├─ models/
│  ├─ best_model.pkl                 # Trained CatBoost model
│  ├─ scaler.pkl                     # StandardScaler object
│  └─ feature_names.pkl              # Feature column names
├─ app.py                             # Streamlit web application
├─ train.py                           # Model training script
├─ requirements.txt                   # Python dependencies
└─ README.md                          # This file
```

---

## 🖥️ Installation & Setup

### 1️⃣ Clone Repository & Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train.py
```

**Output:**
- Trained model saved to `models/best_model.pkl`
- Scaler saved to `models/scaler.pkl`
- Feature names saved to `models/feature_names.pkl`
- Processed data saved to `processed_data/water_potability_cleaned.csv`

### 3️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🔮 Prediction Logic

**Input Flow:**

```
User Input → Feature Engineering → StandardScaler Normalization 
→ CatBoost Prediction → Probability Score → Safety Warnings
```

**Classification Threshold:**

- ✅ **Safe to Drink** — Probability ≥ 0.50  
- ❌ **Not Safe** — Probability < 0.50

**Safety Warnings Triggered:**

- ⚠️ **pH Alert** — Outside range 6.5 – 8.5
- ⚠️ **Turbidity Alert** — Greater than 5 NTU
- ⚠️ **Solids Alert** — Greater than 15,000 mg/L  

---

## 📋 WHO Recommended Safe Ranges

| Parameter | Safe Range |
|-----------|-------------|
| **pH** | 6.5 – 8.5 |
| **Hardness** | 60 – 180 mg/L |
| **Solids (TDS)** | < 1000 mg/L |
| **Chloramines** | 2 – 4 mg/L |
| **Sulfate** | < 250 mg/L |
| **Conductivity** | < 1500 µS/cm |
| **Organic Carbon** | < 5 mg/L |
| **Trihalomethanes** | < 80 µg/L |
| **Turbidity** | < 5 NTU |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|-------------|
| **Python 3.8+** | Core programming language |
| **CatBoost** | Gradient boosting classifier |
| **Scikit-learn** | ML utilities (preprocessing, metrics) |
| **Imbalanced-learn** | SMOTETomek for class imbalancing |
| **Streamlit** | Interactive web UI |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Joblib** | Model serialization |  

---

## 🚀 Future Enhancements

- 🔹 REST API endpoint for integration
- 🔹 Real-time IoT sensor data integration  
- 🔹 Advanced dashboard with visualization charts
- 🔹 Automated test report PDF generation  
- 🔹 Model explainability with SHAP values
- 🔹 Multi-language support  
- 🔹 Model retraining pipeline

---

## 📝 Notes

- **Error Handling:** The app includes try-except blocks to gracefully handle missing model files
- **Reproducibility:** Random seeds set to 42 across all components
- **Data Privacy:** No data is stored; predictions are real-time only

---

## 📧 Contact & Support

**Author:** Bharathi M  
**Field:** AI & Machine Learning  

For questions or issues, please refer to the project repository.  
