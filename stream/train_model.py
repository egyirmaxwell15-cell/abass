import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    "gender": ["male", "female", "female", "male"],
    "parental level of education": [
        "high school",
        "bachelor's degree",
        "some college",
        "master's degree"
    ],
    "test preparation course": ["none", "completed", "completed", "none"],
    "math score": [65, 78, 90, 70]
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=["math score"])
y = df["math score"]

# Encode categorical data
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_encoded, y)

# Save model and encoder together
bundle = {"model": model, "encoder": encoder}
with open("student_score_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("✅ Model trained and saved as 'student_score_model.pkl'")