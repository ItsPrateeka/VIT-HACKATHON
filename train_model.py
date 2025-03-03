import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
df = pd.read_csv("processed_data.csv")

# Drop non-numeric columns
df = df.drop(columns=["Rider_ID", "Date", "Trip_ID", "Charging_Start_Time", "Charging_End_Time"])

# Initialize label encoder
label_enc = LabelEncoder()

# Convert categorical columns to numeric
df["Acceleration_Pattern"] = label_enc.fit_transform(df["Acceleration_Pattern"])
df["Braking_Pattern"] = label_enc.fit_transform(df["Braking_Pattern"])
df["Recommended_Action"] = label_enc.fit_transform(df["Recommended_Action"])

# Print encoded values for verification
print("Encoded Acceleration_Pattern:", df["Acceleration_Pattern"].unique())
print("Encoded Braking_Pattern:", df["Braking_Pattern"].unique())
print("Encoded Recommended_Action:", df["Recommended_Action"].unique())

# Check for missing values in target variables BEFORE splitting
print("\nMissing values BEFORE filling:")
print(df[["Battery_Health_percent", "Estimated_Range_km"]].isnull().sum())

# Fill missing values with mean
df["Battery_Health_percent"] = df["Battery_Health_percent"].fillna(df["Battery_Health_percent"].mean())
df["Estimated_Range_km"] = df["Estimated_Range_km"].fillna(df["Estimated_Range_km"].mean())

# Verify that missing values are gone
print("\nMissing values AFTER filling:")
print(df[["Battery_Health_percent", "Estimated_Range_km"]].isnull().sum())

# Define features (X) and target variables (y)
X = df.drop(columns=["Energy_kWh", "Battery_Health_percent", "Estimated_Range_km", "Recommended_Action"])
y_usage = df["Energy_kWh"]  # Battery usage prediction
y_lifespan = df["Battery_Health_percent"]  # Remaining battery lifespan
y_degradation = df["Estimated_Range_km"]  # Battery degradation trends
y_recommendation = df["Recommended_Action"]  # Charging recommendations (categorical)

# Train-test split (each target variable should have its own split)
X_train_usage, X_test_usage, y_train_usage, y_test_usage = train_test_split(X, y_usage, test_size=0.2, random_state=42)
X_train_lifespan, X_test_lifespan, y_train_lifespan, y_test_lifespan = train_test_split(X, y_lifespan, test_size=0.2, random_state=42)
X_train_degradation, X_test_degradation, y_train_degradation, y_test_degradation = train_test_split(X, y_degradation, test_size=0.2, random_state=42)
X_train_recommendation, X_test_recommendation, y_train_recommendation, y_test_recommendation = train_test_split(X, y_recommendation, test_size=0.2, random_state=42)

# Train models
model_usage = RandomForestRegressor(n_estimators=300, random_state=42)
model_usage.fit(X_train_usage, y_train_usage)

model_lifespan = RandomForestRegressor(n_estimators=300, random_state=42)
model_lifespan.fit(X_train_lifespan, y_train_lifespan)

model_degradation = RandomForestRegressor(n_estimators=300, random_state=42)
model_degradation.fit(X_train_degradation, y_train_degradation)

model_recommendation = RandomForestClassifier(n_estimators=300, random_state=42)
model_recommendation.fit(X_train_recommendation, y_train_recommendation)

# Save models
joblib.dump(model_usage, "battery_usage_model.pkl")
joblib.dump(model_lifespan, "battery_lifespan_model.pkl")
joblib.dump(model_degradation, "battery_degradation_model.pkl")
joblib.dump(model_recommendation, "charging_recommendation_model.pkl")

# Evaluate models
def evaluate_model(y_test, y_pred, metric_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n🔹 {metric_name} Evaluation:")
    print(f"✅ MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

evaluate_model(y_test_usage, model_usage.predict(X_test_usage), "Battery Usage Prediction")
evaluate_model(y_test_lifespan, model_lifespan.predict(X_test_lifespan), "Remaining Battery Lifespan Prediction")
evaluate_model(y_test_degradation, model_degradation.predict(X_test_degradation), "Battery Degradation Prediction")

# Evaluate classification model
y_pred_recommendation = model_recommendation.predict(X_test_recommendation)
accuracy = (y_pred_recommendation == y_test_recommendation).mean()
print(f"\n✅ Charging Recommendation Accuracy: {accuracy:.4f}")
