import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
print("Loading data from data/synthetic_loan_data.csv...")
data = pd.read_csv('data/synthetic_loan_data.csv')
print(f"Data loaded. Shape: {data.shape}\n")

# Overview of the dataset
print("Dataset Overview:")
print(data.describe())
print("\nSample Data:")
print(data.head())

# Split the data into features (X) and target (y)
X = data.drop('approved', axis=1)
y = data['approved']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows\n")

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
print("Training the logistic regression model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
import joblib
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("\nModel saved as 'models/logistic_model.pkl'")
print("Scaler saved as 'models/scaler.pkl'")
