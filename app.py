import joblib
import pandas as pd
import streamlit as st

# Paths
MODEL_PATH = "predictive_maintenance.pkl"
DATA_PATH = r"D:\NITIN\ML_Deployment\CAPSTONE PROJECT 1\CAPSTONE PROJECT 1\Predictive Maintenance\predictive_maintenance.csv"

# Load saved model artifact
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Exact training feature structure columns expected by the model
feature_cols = [
    'Air temperature [K]', 
    'Process temperature [K]',
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]',
    'Type_L', 
    'Type_M'
]

# Streamlit UI
st.title("Predictive Maintenance Dashboard")
st.write("Enter the machine parameters below to predict failure.")

# Create input layout
inputs = {}
cols = st.columns(2)  # Splits inputs into two columns for better UI

# Numerical features that require user input boxes
numerical_cols = [
    'Air temperature [K]', 
    'Process temperature [K]', 
    'Rotational speed [rpm]', 
    'Torque [Nm]', 
    'Tool wear [min]'
]

# FIX: Safely compute numeric medians by filtering out text columns first
try:
    df = pd.read_csv(DATA_PATH)
    # select_dtypes(include='number') drops text columns like 'Product ID' to prevent the error
    numeric_df = df.select_dtypes(include='number') 
    default_values = {col: float(numeric_df[col].median()) for col in numerical_cols}
except Exception:
    # Safe backup values if the CSV path is missing or fails to load
    default_values = {
        'Air temperature [K]': 300.0,
        'Process temperature [K]': 310.0,
        'Rotational speed [rpm]': 1500.0,
        'Torque [Nm]': 40.0,
        'Tool wear [min]': 100.0
    }

# Build numeric input widgets
for i, col in enumerate(numerical_cols):
    with cols[i % 2]:
        inputs[col] = st.number_input(
            label=col,
            value=default_values[col],
            step=0.1 if 'temperature' in col else 1.0
        )

# Handle Categorical Machine Type (Map dropdown options directly to Type_L and Type_M flags)
with cols[len(numerical_cols) % 2]:
    machine_type = st.selectbox("Product Type", options=["Low (L)", "Medium (M)", "High (H)"])

# Assign values based on dropdown selection
if machine_type == "Low (L)":
    inputs['Type_L'] = 1
    inputs['Type_M'] = 0
elif machine_type == "Medium (M)":
    inputs['Type_L'] = 0
    inputs['Type_M'] = 1
else:  # High (H) acts as the baseline reference category
    inputs['Type_L'] = 0
    inputs['Type_M'] = 0

# Prediction logic
if st.button("Predict Machine Status", type="primary"):
    # Enforce exact column arrangement your model expects
    input_df = pd.DataFrame([inputs], columns=feature_cols)
    
    # Generate prediction array
    prediction = model.predict(input_df)
    
    # Display result status banners
    if prediction[0] == 0:
        st.success("Machine Status: **No Failure (Normal Operation)**")
    else:
        st.error("Machine Status: **Failure Detected! Maintenance Required**")
