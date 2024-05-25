import streamlit as st
import numpy as np
import joblib

# Load the saved Decision Tree Regressor model
dt_regressor = joblib.load('decision_tree_regressor.pkl')

# Function for safe division to avoid division by zero
def safe_divide(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))

# Streamlit App
st.title("NPK Prediction App")
st.sidebar.title("Feature Inputs")

# Add a beautiful header image
header_image = "compo.gif"
st.image(header_image, use_column_width=True)

# Sidebar for feature inputs
nitrogen_content = st.sidebar.number_input("Nitrogen Content", min_value=0.0, step=0.5)
phosphorus_content = st.sidebar.number_input("Phosphorus Content", min_value=0.0, step=0.5)
potassium_content = st.sidebar.number_input("Potassium Content", min_value=0.0, step=0.5)
temp_avg = st.sidebar.number_input("Temp_Avg", min_value=0.0, max_value=80.0, step=1.0)

# Calculate ratios
n_p_ratio = safe_divide(nitrogen_content, phosphorus_content)
k_p_ratio = safe_divide(potassium_content, phosphorus_content)
n_k_ratio = safe_divide(nitrogen_content, potassium_content)

# Fill NaN values resulting from division by zero with 0
n_p_ratio = np.nan_to_num(n_p_ratio, nan=0)
k_p_ratio = np.nan_to_num(k_p_ratio, nan=0)
n_k_ratio = np.nan_to_num(n_k_ratio, nan=0)

# Make predictions
if st.sidebar.button("Make Predictions"):
    input_features = np.array([n_p_ratio, k_p_ratio, n_k_ratio, temp_avg]).reshape(1, -1)
    prediction = dt_regressor.predict(input_features)
    st.success(f"Predicted NPK Class based on our model: {prediction[0]}")
    st.toast('Results look well classified', icon='😍')