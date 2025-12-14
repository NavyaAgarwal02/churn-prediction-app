import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load CSV
data = pd.read_csv("churn_data.csv")

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=["Gender", "SubscriptionType"])

# Features
X = data[["Age", "LoginFrequency", "AvgSessionTime", "SupportTickets",
          "Gender_M", "SubscriptionType_Premium", "SubscriptionType_VIP"]]

# FIX: Convert target to numeric
data["Churned"] = data["Churned"].map({"Yes": 1, "No": 0})

# Target - replace 'churn' with your actual column name
y = data["Churned"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")
print("Model trained and saved!")

