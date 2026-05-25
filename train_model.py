import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/car_data.csv")

print(df.head())

# Feature Engineering

# Car age
df["Car_Age"] = 2026 - df["Year"]

# Drop unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)


# Encode categorical columns


le_fuel = LabelEncoder()
le_selling = LabelEncoder()
le_trans = LabelEncoder()

df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le_selling.fit_transform(df["Selling_type"])
df["Transmission"] = le_trans.fit_transform(df["Transmission"])


# Features and Target


X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model Training


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Prediction


y_pred = model.predict(X_test)


# Evaluation


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")


# Save model


joblib.dump(model, "model/car_price_model.pkl")

print("\nModel saved successfully!")