import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load and clean dataset ===
df = pd.read_csv("resumes.csv")
df = df.drop(columns=['Resume_ID', 'Name', 'AI Score (0-100)'])

# Encode categorical columns
label_encoders = {}
for col in ['Skills', 'Education', 'Certifications', 'Job Role']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
df['Recruiter Decision'] = target_encoder.fit_transform(df['Recruiter Decision'])

# Split into features and target
X = df.drop(columns=['Recruiter Decision'])
y = df['Recruiter Decision']

# === Split data into train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# === Save model and encoders ===
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("\n✅ Model and encoders saved successfully!")