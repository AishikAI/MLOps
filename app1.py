from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO

# **Step 1: Automatically Find the Best Model**
def find_latest_model():
    model_files = [f for f in os.listdir() if f.startswith("best_model_") and f.endswith(".pkl")]
    if not model_files:
        return None
    return sorted(model_files, key=os.path.getmtime, reverse=True)[0]  # Load the most recent model

latest_model_file = find_latest_model()

# Ensure that the model is loaded successfully
if latest_model_file:
    model = joblib.load(latest_model_file)
else:
    model = None

# **Step 2: FastAPI Setup**
app = FastAPI()

# **Step 3: Create Input Model for Prediction**
class PredictionInput(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Type: str

# **Failure reasons dictionary**
failure_reasons = {
    "TWF": "Tool Wear Failure - Replace worn-out tools.",
    "HDF": "Heat Dissipation Failure - Improve cooling system.",
    "PWF": "Power Failure - Check electrical components.",
    "OSF": "Overstrain Failure - Reduce excessive force.",
    "RNF": "Random Failure - Perform general maintenance."
}

# **Step 4: Root Route to Check Service Status**
@app.get("/")
def root():
    if model:
        return {"message": f"Predictive Maintenance API is running, using model: {latest_model_file}"}
    else:
        raise HTTPException(status_code=500, detail="No trained model found. Train a model first!")

# **Favicon Route to Handle 404 Errors**
@app.get("/favicon.ico")
def favicon():
    return FileResponse("path_to_your_favicon/favicon.ico")  # Provide the correct path to your favicon.ico file

# **Step 5: Prediction Endpoint**
@app.post("/predict/")
def predict_failure(input_data: PredictionInput):
    if not model:
        raise HTTPException(status_code=500, detail="No model available for prediction.")
    
    # Convert the input data to the expected DataFrame format
    data_dict = input_data.dict()
    input_df = pd.DataFrame([[ 
        data_dict["Air_temperature_K"], 
        data_dict["Process_temperature_K"], 
        data_dict["Rotational_speed_rpm"], 
        data_dict["Torque_Nm"], 
        data_dict["Tool_wear_min"], 
        data_dict["Type"], 0, 0, 0, 0, 0
    ]], columns=["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", 
                 "Torque_Nm", "Tool_wear_min", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"])

    # Make the prediction
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        failure_type = np.random.choice(["TWF", "HDF", "PWF", "OSF", "RNF"])
        result = {
            "prediction": f"⚠️ Machine is at risk of failure due to **{failure_type}**!",
            "remedy": failure_reasons[failure_type]
        }
    else:
        result = {
            "prediction": "✅ Machine is operating normally.",
            "remedy": None
        }
    return result

# **Step 6: Batch Prediction Endpoint**
@app.post("/predict_batch/")
async def predict_batch(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="No model available for batch prediction.")
    
    # Read CSV file and process it
    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    # Ensure that column names in the uploaded data match the expected names
    required_columns = [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
        "Torque [Nm]", "Tool wear [min]"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_columns)}")

    # Rename columns to match model expectations
    df = df.rename(columns={
        "Air temperature [K]": "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]": "Rotational_speed_rpm",
        "Torque [Nm]": "Torque_Nm",
        "Tool wear [min]": "Tool_wear_min"
    })
    
    # Make predictions on the batch data
    predictions = model.predict(df)
    df["Prediction"] = predictions
    df["Prediction"] = df["Prediction"].map({0: "✅ No Failure", 1: "⚠️ Failure Detected"})
    
    return {"predictions": df.to_dict(orient="records")}
