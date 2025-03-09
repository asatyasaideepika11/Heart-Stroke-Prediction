import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Stroke.csv")

# Standardize column names (convert to lowercase and remove spaces)
df.columns = df.columns.str.strip().str.lower()

# Define categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

# Expected categories for each categorical feature (Prevent unseen label errors)
expected_values = {
    "gender": ["Male", "Female", "Other"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Private", "Self-employed", "Govt job", "Children", "Never worked"],
    "residence_type": ["Urban", "Rural"],
    "smoking_status": ["Never smoked", "Formerly smoked", "Smokes", "Unknown"]
}

# Create a dictionary of encoders for each categorical column
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # 🔹 Handle NaN values by replacing them with "Unknown"
    df[col] = df[col].fillna("Unknown")

    # Include both dataset values & expected values
    all_values = list(set(df[col].tolist() + expected_values[col]))
    le.fit(all_values)  # Fit encoder with complete categories
    df[col] = le.transform(df[col])  # Transform dataset
    encoders[col] = le  # Store encoder

# Save the encoders dictionary
with open("vector.pkl", "wb") as file:
    pickle.dump(encoders, file)

print("✅ Encoders saved in 'vector.pkl' with all expected categories.")

# Select features and target
X = df.drop(columns=["id", "stroke"])  # Remove 'id' and 'stroke'

# Debugging: Check if 'id' is still present
y = df["stroke"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE
# Fill missing values (for numerical columns)
X_train.fillna(X_train.median(), inplace=True)  # Replace NaNs with median values
X_test.fillna(X_test.median(), inplace=True)  # Do the same for test data

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model on balanced data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

import numpy as np
print("Class distribution after SMOTE:", np.bincount(y_train_resampled))
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model training complete! Model saved as 'stroke_model.pkl'")
