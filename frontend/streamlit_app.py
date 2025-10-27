# frontend/streamlit_app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  # Ensure backend runs here
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction")

# --- Fetch feature names from backend ---
try:
    response = requests.get(f"{API_URL}/features")
    response.raise_for_status()
    features = response.json().get("features", [])
except Exception:
    st.error("Cannot reach backend. Please start the FastAPI server with `uvicorn main:app --reload`.")
    st.stop()

# --- Feature Inputs ---
st.subheader("Enter Patient Details")

user_data = {}
cols = st.columns(2)

categorical_mappings = {
    "sex": {
        "mapping": {0: "Female", 1: "Male"},
        "help": "Sex: 0 = Female, 1 = Male"
    },
    "chest_pain_type": {
        "mapping": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"},
        "help": "Type of chest pain"
    },
    "fasting_blood_sugar": {
        "mapping": {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
        "help": "Fasting Blood Sugar Level"
    },
    "resting_ecg": {
        "mapping": {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"},
        "help": "Resting Electrocardiographic Results"
    },
    "exercise_induced_angina": {
        "mapping": {0: "No", 1: "Yes"},
        "help": "Exercise-induced angina"
    },
    "st_slope": {
        "mapping": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "help": "ST Slope type"
    },
    "thalassemia": {
        "mapping": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"},
        "help": "Thalassemia type"
    }
}

numeric_ranges = {
    "age": (20, 100, 1),
    "resting_blood_pressure": (80, 250, 1),
    "cholesterol": (100, 600, 1),
    "max_heart_rate": (60, 220, 1),
    "st_depression": (0.0, 6.0, 0.1),
    "num_major_vessels": (0, 4, 1)
}

for i, f in enumerate(features):
    with cols[i % 2]:
        f_lower = f.lower()
        if f_lower in categorical_mappings:
            info = categorical_mappings[f_lower]
            user_data[f] = st.selectbox(
                f,
                options=list(info["mapping"].keys()),
                format_func=lambda x, m=info["mapping"]: m[x],
                help=info["help"]
            )
        else:
            min_val, max_val, step = numeric_ranges.get(f_lower, (0.0, 1000.0, 1.0))
            user_data[f] = st.number_input(
                f,
                value=min_val,
                min_value=min_val,
                max_value=max_val,
                step=step
            )

# --- Prediction Button ---
if st.button("Predict"):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"data": user_data})
        if resp.ok:
            result = resp.json()
            pred = "Heart Disease Detected" if result["prediction"] == 1 else "No Heart Disease"
            st.success(pred)
            st.metric("Probability", f"{result['probability'] * 100:.2f}%")
        else:
            st.error("Prediction failed. " + resp.text)
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
