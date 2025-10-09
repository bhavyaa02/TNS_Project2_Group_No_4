import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("manufacturing_dataset_1000_samples.csv")

# Drop Timestamp column if it exists
if "Timestamp" in df.columns:
    df.drop(columns=["Timestamp"], inplace=True)

# 2️⃣ Handle missing values
df["Material_Viscosity"].fillna(df["Material_Viscosity"].mean(), inplace=True)
df["Ambient_Temperature"].fillna(df["Ambient_Temperature"].mean(), inplace=True)
df["Operator_Experience"].fillna(df["Operator_Experience"].mode()[0], inplace=True)

# 3️⃣ Define features and target
X = df.drop(columns=["Parts_Per_Hour"])
y = df["Parts_Per_Hour"]

# 4️⃣ Identify categorical and numerical columns
cat_cols = ["Shift", "Machine_Type", "Material_Grade", "Day_of_Week"]
num_cols = [col for col in X.columns if col not in cat_cols]

# 5️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

# 6️⃣ Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 7️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8️⃣ Train model
model.fit(X_train, y_train)

# 9️⃣ Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
