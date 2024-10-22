import joblib
import pickle

# Load the existing Joblib model and scaler
model = joblib.load('models/logistic_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Save the model as a Pickle file
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler as a Pickle file
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler successfully converted to Pickle files.")
