import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv("ev-battery-health-100-samples.csv")

# Rename columns for consistency (Optional)
df.rename(columns={"Charging_Type": "Charging Type", "Charging_Station_Location": "Charging Station Location"}, inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
df["Charging Type"] = label_enc.fit_transform(df["Charging Type"])
df["Charging Station Location"] = label_enc.fit_transform(df["Charging Station Location"])

# Normalize numerical features
scaler = MinMaxScaler()
numeric_columns = ["Trip_Distance_km", "Trip_Duration_min", "Average_Speed_kmph", 
                   "Voltage_V", "Current_A", "Energy_kWh", "Charge_Cycles",
                   "Depth_of_Discharge_percent", "Temperature_C", "Battery_Charge_Efficiency_percent"]
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save preprocessed dataset
df.to_csv("processed_data.csv", index=False)
print("Dataset Preprocessed & Saved!")
