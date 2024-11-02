# Loan Model Training

A machine learning project for training a loan approval prediction model using synthetic data. This project includes data generation, model training, and evaluation components.

## Overview

This project implements a loan approval prediction system using logistic regression. It consists of two main components:
- Synthetic data generation with realistic loan application parameters
- Model training and evaluation pipeline

## Requirements

The project requires Python 3.x and the following dependencies:

python
pandas==1.5.3
flask==2.2.5
flask_sqlalchemy==2.5.1
flask-cors==3.0.10
numpy==1.26.0
psycopg2-binary==2.9.10
scikit-learn==1.3.2
werkzeug==2.2.3
scipy==1.11.4
SQLAlchemy<2.0


## Project Structure

oan-model-training/
├── data/
│ └── synthetic_loan_data.csv
├── models/
│ ├── logistic_model.pkl
│ └── scaler.pkl
├── datamaker.py
├── train_model.py
├── requirements.txt
└── README.md



## Features

The model considers the following features for loan approval prediction:
- Income
- Loan amount
- Credit history
- Debt-to-income ratio
- Loan term
- Employment length
- Loan-to-income ratio

## Usage

1. First, generate synthetic data:

bash
python datamaker.py



2. Train the model:


## Data Generation

The synthetic data generator (`datamaker.py`) creates realistic loan application data with the following characteristics:
- 50,000 sample records
- Realistic income and loan amount distributions
- Credit history binary classification
- Various loan terms (12, 24, 36, 48, 60 months)
- Employment length between 0-30 years
- Realistic debt-to-income and loan-to-income ratios

## Model Training

The training script (`train_model.py`) performs the following:
- Loads and preprocesses the synthetic data
- Splits data into training and test sets (80/20)
- Applies feature scaling
- Trains a logistic regression model
- Evaluates model performance
- Saves the trained model and scaler for future use

## Model Output

The trained model and scaler are saved as:
- `models/logistic_model.pkl`: The trained logistic regression model
- `models/scaler.pkl`: The fitted StandardScaler object

## Contributing

Feel free to submit issues and enhancement requests!

