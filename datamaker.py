import pandas as pd
import numpy as np

# Parameters reflecting current economic trends
np.random.seed(42)
num_samples = 50000

# Generate synthetic data aligned with underwriting practices
data = pd.DataFrame({
    'income': np.random.normal(60000, 15000, num_samples).clip(20000, 130000),
    'loan_amount': np.random.normal(30000, 10000, num_samples).clip(5000, 70000),
    'credit_history': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),
    'debt_to_income': np.random.uniform(0.1, 0.6, num_samples),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], size=num_samples, p=[0.1, 0.15, 0.4, 0.2, 0.15]),
    'employment_length': np.random.randint(0, 30, num_samples),
    'loan_to_income': np.random.uniform(0.05, 3, num_samples),
})

# Generate 'approved' based on realistic approval logic
data['approved'] = np.where(
    (data['credit_history'] == 1) & 
    (data['debt_to_income'] < 0.4) & 
    (data['employment_length'] >= 2) & 
    (data['loan_to_income'] < 1), 1, 0
)

# Save the data
data.to_csv('data/synthetic_loan_data.csv', index=False)
print(f"Generated {num_samples} rows of data and saved to 'data/synthetic_loan_data.csv'.")
