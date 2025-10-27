# 🫀 Heart Disease Prediction  
### TNS Project 2 – Group No. 4  

---

## 📘 Overview  
This project predicts the likelihood of **heart disease** using a **machine learning model** built in Python.  
It includes:  
- A **FastAPI backend** for prediction APIs  
- A **Streamlit frontend** for an interactive user interface  

---

## ⚙️ Tech Stack  
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **ML Libraries:** Scikit-learn, Pandas, NumPy  
- **Server:** Uvicorn  
- **Version Control:** Git & GitHub  

---

## 🗂️ Project Structure  
```text
TNS_Project2_Group_No_4/
│
├── backend/
│   ├── main.py
│   ├── models/
│   │   ├── best_heart_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_names.json
│
├── frontend/
│   └── streamlit_app.py
│
└── README.md


##🚀 How to Run
1️⃣ Clone the Repository
git clone https://github.com/bhavyaa02/TNS_Project2_Group_No_4.git
cd TNS_Project2_Group_No_4

2️⃣ Run Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload


➡️ Backend will start at: http://127.0.0.1:8000

3️⃣ Run Frontend (Streamlit)

Open a new terminal window:

cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py


➡️ Frontend will start at: http://localhost:8501

🔮 Future Enhancements

Add SHAP/LIME explainability

Add patient data logging
