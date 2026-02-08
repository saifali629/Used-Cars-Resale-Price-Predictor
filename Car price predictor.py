# 0) UPLOAD DATASET FROM YOUR PC
df = pd.read_excel("used_car_dataset.xlsx")


# 1) IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

# 2) LOAD DATASET
df = pd.read_excel("used_car_dataset.xlsx")
print("Dataset Loaded Successfully!")

# 3) BASIC CLEANING
df = df.dropna(subset=["Resale_Price"])

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# 4) SELECT FEATURES
features = [
    "Model_Year",
    "Mileage",
    "Engine_CC",
    "Transmission",
    "Fuel_Type",
    "Condition_Score",
    "Accident_History",
    "Region",
    "Market_Demand",
    "Base_New_Price"
]

X = df[features]
y = df["Resale_Price"]

# 5) TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) PREPROCESSING
categorical = X.select_dtypes(include="object").columns.tolist()
numeric = X.select_dtypes(include=np.number).columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

# 7) TRAIN MODELS
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess),
                     ("model", model)])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    results[name] = (mae, rmse, r2, pipe)

# 8) SELECT BEST MODEL
best_model = min(results.items(), key=lambda x: x[1][1])
best_name = best_model[0]
best_pipe = best_model[1][3]
print("\nBEST MODEL =", best_name)


# 8.1) SHOW SAMPLE PREDICTIONS (TEST DATA)

test_predictions = best_pipe.predict(X_test)

comparison = pd.DataFrame({
    "Actual_Resale_Price": y_test.values,
    "Predicted_Resale_Price": test_predictions
})

print("\nSample Predictions (Test Data):")
print(comparison.head(10))

# 9) SAVE RESULTS
df["Predicted_Resale_Price"] = best_pipe.predict(X)
df.to_csv("used_car_predictions.csv", index=False)

joblib.dump(best_pipe, f"{best_name}_model.joblib")

print("\nSaved:")
print("used_car_predictions.csv")
print(f"{best_name}_model.joblib")


# 10) SIMPLE GRAPH WITH REGRESSION LINE

y_pred = best_pipe.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")

# Regression line (ideal line)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--',
    label="Regression Line"
)

plt.xlabel("Actual Resale Price")
plt.ylabel("Predicted Resale Price")
plt.title(f"Actual vs Predicted ({best_name})")
plt.legend()
plt.show()
