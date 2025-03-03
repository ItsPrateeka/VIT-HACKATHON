from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
model_usage = joblib.load("battery_usage_model.pkl")
model_lifespan = joblib.load("battery_lifespan_model.pkl")
model_degradation = joblib.load("battery_degradation_model.pkl")
model_recommendation = joblib.load("charging_recommendation_model.pkl")

# Define the features required by the model
FEATURE_COLUMNS = [
    "Trip_Distance_km", "Trip_Duration_min", "Average_Speed_kmph",
    "Acceleration_Pattern", "Braking_Pattern", "Energy_Consumption_kWh_per_km",
    "Idle_Time_min", "Voltage_V", "Current_A", "Charge_Cycles",
    "Depth_of_Discharge_percent", "Temperature_C", "Charging_Power_kW",
    "Charging_Duration_min", "Charging Type", "Charging Station Location",
    "Battery_Charge_Efficiency_percent"
]

@app.route('/')
def home():
    return "EV Battery Health Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data
        data = request.get_json()
        print("Received Input:", data)  # Print input data for debugging
        
        # Convert input data into a DataFrame
        input_data = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Ensure data is in correct format
        input_data = input_data.astype(float)

        # Make predictions
        battery_usage_pred = model_usage.predict(input_data)[0]
        lifespan_pred = model_lifespan.predict(input_data)[0]
        degradation_pred = model_degradation.predict(input_data)[0]
        recommendation_pred = model_recommendation.predict(input_data)[0]

        # Return predictions as JSON response
        return jsonify({
            "Predicted Battery Usage (kWh)": round(float(battery_usage_pred), 4),
            "Predicted Battery Lifespan (%)": round(float(lifespan_pred), 2),
            "Predicted Battery Degradation (km)": round(float(degradation_pred), 2),
            "Optimal Charging Recommendation": int(recommendation_pred)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
