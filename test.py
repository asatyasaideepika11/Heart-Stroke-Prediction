import pandas as pd

# Load dataset
df = pd.read_csv("Stroke.csv")

# Count how many "stroke" cases exist
print(df["Stroke"].value_counts())
