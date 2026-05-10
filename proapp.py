# AI-Based Health Risk Prediction (Final Version - Local CSV)

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
# Replace 'health_data.csv' with your file name
data = pd.read_csv("health_data.csv")  

# Step 3: Explore Data
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nDataset Statistics:")
print(data.describe())

# Step 4: Preprocessing
X = data.drop('target', axis=1)  # features
y = data['target']               # labels

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Predict a new sample patient
# Replace these values with real patient data
sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)
print("\nPredicted Health Risk (1 = disease, 0 = no disease):", prediction[0])