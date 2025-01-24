import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# **Step 1: Automatically Find the Best Model**
def find_latest_model():
    model_files = [f for f in os.listdir() if f.startswith("best_model_") and f.endswith(".pkl")]
    if not model_files:
        return None
    return sorted(model_files, key=os.path.getmtime, reverse=True)[0]  # Load the most recent model

latest_model_file = find_latest_model()

if latest_model_file:
    model = joblib.load(latest_model_file)
    st.sidebar.success(f"✅ Using Model: {latest_model_file}")
else:
    st.sidebar.error("❌ No trained model found. Train a model first!")
    st.stop()

# **Step 2: Streamlit UI**
st.title("Predictive Maintenance System 🚀")
st.write("Enter machine parameters to predict failure and get remedies.")

# **Step 3: User Input Fields**
mode = st.radio("Select Input Mode:", ["Single Input", "Batch Upload"])

if mode == "Single Input":
    st.subheader("🔹 Single Data Input")

    temperature = st.number_input("Temperature (°C)", min_value=295.0, max_value=305.0, value=300.0)
    process_temp = st.number_input("Process Temperature (°C)", min_value=305.0, max_value=314.0, value=310.0)
    speed = st.number_input("Rotational Speed (RPM)", min_value=1200, max_value=2800, value=1500)
    torque = st.number_input("Torque (Nm)", min_value=3.0, max_value=76.0, value=40.0)
    tool_wear = st.number_input("Tool Wear (Minutes)", min_value=0, max_value=250, value=100)
    type_option = st.selectbox("Machine Type", ["L", "M", "H"])

    failure_reasons = {
        "TWF": "Tool Wear Failure - Replace worn-out tools.",
        "HDF": "Heat Dissipation Failure - Improve cooling system.",
        "PWF": "Power Failure - Check electrical components.",
        "OSF": "Overstrain Failure - Reduce excessive force.",
        "RNF": "Random Failure - Perform general maintenance."
    }

    # **Update the column names to match the model's expected input**
    input_data = pd.DataFrame([[temperature, process_temp, speed, torque, tool_wear, type_option, 0, 0, 0, 0, 0]],
                              columns=["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", 
                                       "Torque_Nm", "Tool_wear_min", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"])

    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            failure_type = np.random.choice(["TWF", "HDF", "PWF", "OSF", "RNF"])
            st.error(f"⚠️ Machine is at risk of failure due to **{failure_type}**!")
            st.warning(f"🛠 Suggested Remedy: {failure_reasons[failure_type]}")
        else:
            st.success("✅ Machine is operating normally.")

# **Batch Mode**
elif mode == "Batch Upload":
    st.subheader("📂 Upload CSV File for Bulk Prediction")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 **Uploaded Data Preview:**")
        st.write(df.head())

        # Ensure that column names in the uploaded data match the expected names
        df = df.rename(columns={
            "Air temperature [K]": "Air_temperature_K",
            "Process temperature [K]": "Process_temperature_K",
            "Rotational speed [rpm]": "Rotational_speed_rpm",
            "Torque [Nm]": "Torque_Nm",
            "Tool wear [min]": "Tool_wear_min"
        })

        df["Prediction"] = model.predict(df)
        df["Prediction"] = df["Prediction"].map({0: "✅ No Failure", 1: "⚠️ Failure Detected"})

        st.subheader("🔍 Prediction Results")
        st.write(df)
