import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load the optimized dataset
df = pd.read_csv('data/synthetic_loan_data.csv')

# Define features and target variable
X = df[['applicant_income', 'loan_amount', 'credit_history', 
        'debt_to_income', 'loan_term', 'employment_length', 'loan_to_income']]
y = df['approved']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Solvers for small and large datasets
}

grid = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Use the best model from GridSearchCV
model = grid.best_estimator_

# Evaluate the model using the test set
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC score
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Save the model and scaler
joblib.dump(model, 'models/logistic_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("Model and scaler saved in the 'models/' directory")
