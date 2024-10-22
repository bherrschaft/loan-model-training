import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate realistic applicant income (30,000 to 200,000 USD)
applicant_income = np.random.randint(30000, 200000, 1000)

# Loan amount proportional to income (10% to 40%)
loan_amount = np.round(applicant_income * np.random.uniform(0.1, 0.4, 1000), 2)

# Credit history (1 = Good, 0 = Bad); Higher income => Better credit history
credit_history = np.where(applicant_income > 75000, 1, np.random.choice([0, 1], 1000, p=[0.5, 0.5]))

# DTI decreases slightly with higher income; clipped for realism
debt_to_income = np.clip(np.random.uniform(0.1, 0.5, 1000) - (applicant_income / 500000), 0.1, 0.7)

# Loan term: 12, 24, 36, or 60 months; higher income => shorter loan term
loan_term = np.where(applicant_income > 100000, np.random.choice([12, 24], 1000), np.random.choice([36, 60], 1000))

# Employment length: Higher incomes correlate with longer employment
employment_length = np.where(applicant_income > 75000, np.random.randint(5, 20, 1000), np.random.randint(1, 15, 1000))

# Loan-to-income ratio: Loan amount divided by income
loan_to_income = loan_amount / applicant_income

# Approval logic: Realistic underwriting conditions
approved = np.where(
    (credit_history == 1) & (debt_to_income < 0.4) & (employment_length >= 3) & (loan_to_income < 0.4), 
    1, 
    0
)

# Create DataFrame
data = {
    'applicant_income': applicant_income,
    'loan_amount': loan_amount,
    'credit_history': credit_history,
    'debt_to_income': debt_to_income,
    'loan_term': loan_term,
    'employment_length': employment_length,
    'loan_to_income': loan_to_income,
    'approved': approved
}

# Save dataset to CSV
df = pd.DataFrame(data)
df.to_csv('data/synthetic_loan_data.csv', index=False)

print("Realistic synthetic loan data saved to 'data/synthetic_loan_data.csv'")
