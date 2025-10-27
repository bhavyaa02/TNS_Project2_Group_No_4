
## 🧠 Capstone Project 1 — Machine Learning Web Application

### 📘 Overview

This project is a full-stack **Machine Learning Web Application** that demonstrates the complete ML lifecycle — from data preprocessing and model training to serving predictions through a Flask-based frontend.

The project aims to build a predictive model (e.g., heart disease or manufacturing defect detection) and deploy it as a user-friendly web app.

---

### 🏗️ Project Structure

```
Capstone_project_1/
├── backend/
│   ├── main.py                # Backend Flask app or API server
│   ├── model.pkl              # Trained ML model
│
├── frontend/
│   └── app.py                 # Frontend Flask entry point / UI integration
│
├── data/
│   └── raw/
│       └── heart_disease_dataset.csv  # Original dataset used for training
│
├── model/
│   ├── train_model.py         # Script for model training
│   ├── model.pkl              # Trained model
│   └── manufacturing_dataset_1000_samples.csv
│
├── src/
│   ├── config.py              # Configuration constants
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── train.py               # Model training logic
│   ├── evaluate.py            # Model evaluation and metrics computation
│   ├── metrics.py             # Evaluation metric functions
│   ├── serve.py               # Script to serve model via Flask API
│   └── utils.py               # Utility functions
│
├── reports/
│   └── model_comparison.csv   # Comparison results of multiple trained models
│
├── requirements.txt           # All required dependencies
└── README.md                  # (This file)
```

---

### ⚙️ Features

✅ End-to-end machine learning workflow
✅ Modular Python scripts for training and evaluation
✅ Flask-based backend for real-time predictions
✅ Interactive web frontend
✅ Data preprocessing and metrics tracking
✅ Easy deployment setup

---

### 🧩 Tech Stack

| Category            | Tools Used                          |
| ------------------- | ----------------------------------- |
| **Language**        | Python 3                            |
| **Web Framework**   | Flask                               |
| **ML / Data**       | scikit-learn, pandas, numpy         |
| **Visualization**   | matplotlib / seaborn                |
| **Frontend**        | HTML, CSS, JS (via Flask templates) |
| **Version Control** | Git & GitHub                        |

---

### 🧠 Model Workflow

1. **Data Preprocessing:** Cleaning and encoding data in `src/preprocessing.py`.
2. **Training:** Model trained via `model/train_model.py` or `src/train.py`.
3. **Evaluation:** Performance recorded in `src/evaluate.py` and saved in `reports/model_comparison.csv`.
4. **Serialization:** The best-performing model saved as `model.pkl`.
5. **Deployment:** Flask app (`backend/main.py`) loads the model and serves predictions to the frontend.

---

### 🖥️ How to Run Locally

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bhavyaa02/TNS_Project2_Group_No_4.git
cd TNS_Project2_Group_No_4
```

#### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run the Backend

```bash
cd backend
python main.py
```

#### 5️⃣ Run the Frontend

```bash
cd ../frontend
python app.py
```

Then open your browser at:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

### 📊 Example Output

Once running, the web app lets users:

* Upload or input data values,
* Get instant model predictions, and
* View metrics and results in an interactive UI.




