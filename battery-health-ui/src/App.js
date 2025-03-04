import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:5000";

const App = () => {
  const [formData, setFormData] = useState({
    Trip_Distance_km: "",
    Trip_Duration_min: "",
    Average_Speed_kmph: "",
    Acceleration_Pattern: "",
    Braking_Pattern: "",
    Energy_Consumption_kWh_per_km: "",
    Idle_Time_min: "",
    Voltage_V: "",
    Current_A: "",
    Charge_Cycles: "",
    Depth_of_Discharge_percent: "",
    Temperature_C: "",
    Charging_Power_kW: "",
    Charging_Duration_min: "",
    Charging_Type: "",
    Charging_Station_Location: "",
    Battery_Charge_Efficiency_percent: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null); // Reset errors before submitting

    try {
      const requestData = Object.fromEntries(
        Object.entries(formData).map(([key, value]) => [key, parseFloat(value)])
      );

      const response = await axios.post(`${API_URL}/predict`, requestData);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error making prediction:", error);
      setError("Failed to get prediction. Please check your input values and try again.");
    }
};

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>EV Battery Health Monitoring</h2>

      <form onSubmit={handleSubmit} style={styles.form}>
        {Object.keys(formData).map((key) => (
          <input
            key={key}
            type="number"
            name={key}
            placeholder={key.replace(/_/g, " ")}
            value={formData[key]}
            onChange={handleChange}
            required
            style={styles.input}
          />
        ))}
        <button type="submit" style={styles.button}>Predict</button>
      </form>

      {error && <p style={styles.error}>{error}</p>}

      {prediction && (
        <div style={styles.resultBox}>
          <h3 style={styles.resultTitle}>Prediction Results:</h3>
          <p><strong>Battery Usage (kWh):</strong> {prediction["Predicted Battery Usage (kWh)"]}</p>
          <p><strong>Battery Lifespan (%):</strong> {prediction["Predicted Battery Lifespan (%)"]}</p>
          <p><strong>Battery Degradation (km):</strong> {prediction["Predicted Battery Degradation (km)"]}</p>
          <p><strong>Charging Recommendation:</strong> {prediction["Optimal Charging Recommendation"]}</p>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: { textAlign: "center", padding: "20px", fontFamily: "Arial, sans-serif" },
  title: { color: "#2C3E50" },
  form: {
    display: "grid",
    gap: "10px",
    width: "50%",
    margin: "auto",
    padding: "20px",
    border: "1px solid #ccc",
    borderRadius: "10px",
    backgroundColor: "#f9f9f9"
  },
  input: {
    padding: "10px",
    borderRadius: "5px",
    border: "1px solid #ddd",
    fontSize: "14px"
  },
  button: {
    backgroundColor: "#3498db",
    color: "white",
    padding: "10px",
    border: "none",
    borderRadius: "5px",
    fontSize: "16px",
    cursor: "pointer"
  },
  error: { color: "red", marginTop: "10px" },
  resultBox: {
    marginTop: "20px",
    padding: "15px",
    border: "1px solid #2ecc71",
    borderRadius: "10px",
    backgroundColor: "#e8f5e9",
    width: "50%",
    margin: "auto"
  },
  resultTitle: { color: "#27ae60" }
};

export default App;
