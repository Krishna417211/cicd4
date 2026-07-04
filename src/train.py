loan_data = {
    "age": [22, 25, 30, 35, 40, 28, 45, 32, 24, 38, 29, 50],
    "income": [20000, 25000, 50000, 65000, 80000, 30000, 90000, 55000, 22000, 70000, 42000, 100000],
    "loan_amount": [100000, 120000, 250000, 300000, 400000, 150000, 500000, 280000, 110000, 350000, 220000, 600000],
    "credit_score": [550, 600, 720, 750, 780, 640, 800, 710, 580, 770, 690, 820],
    "loan_approved": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}


import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Convert dictionary data into DataFrame
df = pd.DataFrame(loan_data)

# Input features
X = df[["age", "income", "loan_amount", "credit_score"]]

# Target column
y = df["loan_approved"]

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Minimum model quality check
if accuracy < 0.60:
    raise ValueError("Model accuracy is below the required threshold.")

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('loan_experiment')


with mlflow.start_run():
    mlflow.log_param('max_iter', 1000)
    mlflow.log_metric('accuracy',accuracy)
   
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="loan_model",
        registered_model_name="loan_model"
)
    print("logged in")


joblib.dump(model,'models/loan_model.pkl')
    