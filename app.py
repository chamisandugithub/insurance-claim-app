import streamlit as st
import pandas as pd
import joblib
import base64

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Insurance Claim Predictor",
    layout="centered"
)

# -----------------------------------
# Background image
# -----------------------------------
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("Blog_Insurance-Claim-Processing-Simple.jpg")

# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown("""
<style>
[data-testid="stForm"] {
    background-color: transparent;
    padding: 20px;
    border-radius: 10px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Title
# -----------------------------------
st.title(" Insurance Claim Prediction App")
st.write("Enter customer details to predict insurance claim amount.")

# -----------------------------------
# Load trained model (FIXED NAME)
# -----------------------------------
model = joblib.load("best_model.pkl")

# -----------------------------------
# Input form
# -----------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
        diabetic = st.selectbox("Diabetic", ["Yes", "No"])
        region = st.selectbox(
            "Region",
            ["southeast", "southwest", "northwest", "northeast"]
        )

    with col2:
        bloodpressure = st.number_input("Blood Pressure", min_value=60, max_value=180, value=90)
        gender = st.selectbox("Gender", ["male", "female"])
        smoker = st.selectbox("Smoker", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Claim")

# -----------------------------------
# Prediction
# -----------------------------------
if submitted:
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"**Estimated Insurance Claim Amount:** ${prediction:,.2f}")
