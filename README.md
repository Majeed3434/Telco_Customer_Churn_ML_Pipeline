# Telco Customer Churn Pipeline

## Objective
Build a production-ready ML pipeline to predict customer churn using Scikit-Learn.

## Methodology
- **Preprocessing:** `ColumnTransformer` handles numerical scaling and categorical One-Hot Encoding.
- **Model:** Random Forest Classifier.
- **Tuning:** `GridSearchCV` for optimal hyperparameters.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python churn_pipeline.py`
3. Run a prediction: `python app.py`

## Dataset
Telco Customer Churn Dataset 
(https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn)

## Methodology
- Data preprocessing using ColumnTransformer
- Feature scaling & encoding via Pipeline
- Logistic Regression & Random Forest models
- Hyperparameter tuning with GridSearchCV
- Model export using joblib

## Results
Random Forest achieved the best performance based on F1-score.

## Output
Reusable pipeline saved as `churn_pipeline.joblib`

