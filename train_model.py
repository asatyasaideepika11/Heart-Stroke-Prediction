import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# ✅ Load the Framingham dataset
df = pd.read_csv("framingham.csv")  # Make sure the dataset is in the same directory

# ✅ Drop unnecessary columns (if any)
df.drop(columns=["education"], inplace=True, errors="ignore")  # Education is not a strong predictor

# ✅ Handle missing values
df.fillna(df.median(), inplace=True)  # Replace NaN with median values

# ✅ Define features (X) and target (y)
X = df.drop(columns=["TenYearCHD"])  # 'TenYearCHD' is the target variable (stroke risk)
y = df["TenYearCHD"]

# ✅ Normalize numerical features
scaler = StandardScaler()
X[["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = scaler.fit_transform(
    X[["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]]
)

# ✅ Balance the dataset using SMOTE (if needed)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Train the model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# ✅ Save the trained model
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ✅ Save the scaler for use in `app.py`
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model training complete! Model saved as 'stroke_model.pkl'")
