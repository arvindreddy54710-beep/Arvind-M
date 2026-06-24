import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. MLflow Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Home_Credit_AMT_CREDIT_Regression")

registered_model_name = "Home_Credit_AMT_CREDIT"

# 2. Load Dataset
print("Loading data...")
df = pd.read_csv(r'c:\Users\Manjula\Downloads\home-credit-default-risk\application_train.csv')

features = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_POPULATION_RELATIVE']
df = df.dropna(subset=features + ['AMT_CREDIT'])

# Sample 5000 rows to speed up training
df = df.sample(n=5000, random_state=42)

X = df[features]
y = df['AMT_CREDIT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train and Save 10 Model Versions
for version in range(1, 11):
    run_name = f"Random Forest_version_{version}"
    with mlflow.start_run(run_name=run_name):
        n_estimators = 10 + (version * 5)
        max_depth = 5 + (version % 3)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        mlflow.log_param("version", version)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        
        # Log and Register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        print(f"Model {run_name} saved with RMSE: {rmse:.2f}")

print("Done! You can now check the MLflow UI.")
