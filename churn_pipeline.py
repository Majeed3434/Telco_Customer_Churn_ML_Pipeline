import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save():
    # Load and prep
    df = pd.read_csv('data/telco_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Preprocessing
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Model Comparison
    models = {
        'LogisticRegression': (LogisticRegression(), {'classifier__C': [0.1, 1, 10]}),
        'RandomForest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100]})
    }

    best_models = {}
    for name, (model, params) in models.items():
        pipe = Pipeline([('pre', preprocessor), ('classifier', model)])
        grid = GridSearchCV(pipe, params, cv=3).fit(X, y)
        best_models[name] = grid.best_estimator_
        print(f"{name} Best Score: {grid.best_score_:.4f}")

    # Export both
    joblib.dump(best_models, 'models_bundle.joblib')
    print("Models saved to 'models_bundle.joblib'")

if __name__ == "__main__":
    train_and_save()