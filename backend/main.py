from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize app
app = FastAPI(title="Manufacturing Output Prediction API")

# Load model
model = joblib.load("model.pkl")

# Define input schema
class MachineData(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float

@app.post("/predict")
def predict_output(data: MachineData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_parts_per_hour": prediction}
