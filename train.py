# ===============================================
# ✅ TRAIN WATER POTABILITY MODEL (FINAL VERSION)
# ===============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTETomek
from catboost import CatBoostClassifier
import joblib
import os

# ------------------------------------------------
# ✅ Load dataset
# ------------------------------------------------
df = pd.read_csv("data/water_potability.csv")
print("✅ Dataset Loaded:", df.shape)

# ------------------------------------------------
# ✅ Handle Missing Values
# ------------------------------------------------
df.fillna(df.median(numeric_only=True), inplace=True)

# ------------------------------------------------
# ✅ Feature Engineering
# ------------------------------------------------
df["ph_diff_from_7"] = abs(df["ph"] - 7)
df["toxicity_mul"] = df["Chloramines"] * df["Trihalomethanes"]
df["high_turbidity_flag"] = (df["Turbidity"] > 4).astype(int)

# Save processed dataset
os.makedirs("processed_data", exist_ok=True)
df.to_csv("processed_data/water_potability_cleaned.csv", index=False)

# ------------------------------------------------
# ✅ Split Features & Target
# ------------------------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------
# ✅ Scaling
# ------------------------------------------------
num_cols = X_train.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# ✅ Balance Data (SMOTETomek)
# ------------------------------------------------
smk = SMOTETomek(random_state=42)
X_resampled, y_resampled = smk.fit_resample(X_train_scaled, y_train)

# ------------------------------------------------
# ✅ Train CatBoost Model
# ------------------------------------------------
model = CatBoostClassifier(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    loss_function="CrossEntropy",
    random_seed=42,
    verbose=200
)

model.fit(X_resampled, y_resampled)

# ------------------------------------------------
# ✅ Evaluate
# ------------------------------------------------
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------------------------
# ✅ Save Artifacts
# ------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(num_cols, "models/feature_names.pkl")

print("✅ Training Complete — Model, Scaler, Features Saved!")
