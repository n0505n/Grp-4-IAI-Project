# AI-Based Health Risk Prediction
# Educational Machine Learning Project

# Install required libraries before running:
# pip install pandas scikit-learn tensorflow matplotlib seaborn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------------------

# Example:
# Replace 'health_data.csv' with your Kaggle dataset file

df = pd.read_csv("health_data.csv")

print("First 5 rows:")
print(df.head())

# ---------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------

# Remove missing values
df = df.dropna()

# Example categorical columns
categorical_cols = ['Smoking', 'FamilyHistory', 'Diet']

# Encode categorical data
encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Features and target
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# Encode target labels
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# STEP 3: MACHINE LEARNING MODEL (SCIKIT-LEARN)
# ---------------------------------------------------

print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

# ---------------------------------------------------
# STEP 4: TENSORFLOW NEURAL NETWORK
# ---------------------------------------------------

print("\nTraining TensorFlow Neural Network...")

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')   # Example: Low, Moderate, High risk
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTensorFlow Model Accuracy:")
print(accuracy)

# ---------------------------------------------------
# STEP 5: PREDICT NEW PATIENT DATA
# ---------------------------------------------------

# Example new patient:
new_patient = np.array([[
    45,     # Age
    29.5,   # BMI
    140,    # Blood Pressure
    160,    # Glucose
    1,      # Smoking
    1,      # FamilyHistory
    0       # Diet
]])

# Scale input
new_patient = scaler.transform(new_patient)

# Prediction
prediction = rf_model.predict(new_patient)

risk_labels = ["Low", "Moderate", "High"]

print("\nPredicted Health Risk:")
print(risk_labels[prediction[0]])

# ---------------------------------------------------
# RESPONSIBLE AI NOTE
# ---------------------------------------------------

print("\nNOTE:")
print("This project is for educational purposes only.")
print("It is NOT a real medical diagnosis system.")