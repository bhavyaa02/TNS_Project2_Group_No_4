import streamlit as st
import requests
import pandas as pd

st.title("🏭 Manufacturing Output Prediction")
st.markdown("Predict hourly parts output based on machine parameters.")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    Injection_Temperature = st.number_input("Injection Temperature", 0.0)
    Injection_Pressure = st.number_input("Injection Pressure", 0.0)
    Cycle_Time = st.number_input("Cycle Time", 0.0)
    Cooling_Time = st.number_input("Cooling Time", 0.0)
    Material_Viscosity = st.number_input("Material Viscosity", 0.0)
    Ambient_Temperature = st.number_input("Ambient Temperature", 0.0)
   

with col2:
    Shift = st.selectbox("Shift", ["Day", "Night", "Evening"])
    Machine_Type = st.selectbox("Machine Type", ["Type_A", "Type_B"])
    Material_Grade = st.selectbox("Material Grade", ["Economy", "Standard", "Premium"])
    Day_of_Week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    Temperature_Pressure_Ratio = st.number_input("Temperature Pressure Ratio", 0.0)
   
    
with col3:
    Machine_Age = st.number_input("Machine Age", 0.0)
    Operator_Experience = st.number_input("Operator Experience", 0.0)
    Maintenance_Hours = st.number_input("Maintenance Hours", 0.0)
    Total_Cycle_Time = st.number_input("Total Cycle Time", 0.0)
    Efficiency_Score = st.number_input("Efficiency Score", 0.0)
    Machine_Utilization = st.number_input("Machine Utilization", 0.0)
    

if st.button("Predict"):
    input_data = {
        "Injection_Temperature": Injection_Temperature,
        "Injection_Pressure": Injection_Pressure,
        "Cycle_Time": Cycle_Time,
        "Cooling_Time": Cooling_Time,
        "Material_Viscosity": Material_Viscosity,
        "Ambient_Temperature": Ambient_Temperature,
        "Machine_Age": Machine_Age,
        "Operator_Experience": Operator_Experience,
        "Maintenance_Hours": Maintenance_Hours,
        "Shift": Shift,
        "Machine_Type": Machine_Type,
        "Material_Grade": Material_Grade,
        "Day_of_Week": Day_of_Week,
        "Temperature_Pressure_Ratio": Temperature_Pressure_Ratio,
        "Total_Cycle_Time": Total_Cycle_Time,
        "Efficiency_Score": Efficiency_Score,
        "Machine_Utilization": Machine_Utilization
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    if response.status_code == 200:
        prediction = response.json()["predicted_parts_per_hour"]
        st.success(f"✅ Predicted Hourly Output: {prediction:.2f} parts/hour")
    else:
        st.error("❌ Error connecting to the backend.")
