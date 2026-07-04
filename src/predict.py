import joblib
import pandas as pd



# Load trained model
model = joblib.load("models/loan_model.pkl")


# New customer data
new_customer = {
    "age": [31],
    "income": [60000],
    "loan_amount": [250000],
    "credit_score": [730]
}


# Convert dictionary into DataFrame
customer_df = pd.DataFrame(new_customer)


# Predict loan approval
prediction = model.predict(customer_df)[0]

# Predict approval probability
probability = model.predict_proba(customer_df)[0][1]


# Display result
if prediction == 1:
    print("Loan Status: Approved")
else:
    print("Loan Status: Rejected")

print("Approval Probability:", round(probability, 2))