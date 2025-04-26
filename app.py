import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# --- Project Description ---
st.title("Who Gets Paid More? – Interactive Income Prediction App")

st.markdown("""
This app predicts whether an individual is likely to earn more than $50,000 per year based on US Census demographic and employment data. 
It uses a Random Forest model trained on the UCI Adult dataset to deliver accurate, real-world predictions for a wide range of scenarios. 
Enter details below to see a live prediction and understand the key factors that impact income.

**Project by [Dhairya Negandhi](https://www.linkedin.com/in/dhairya-negandhi-933b381bb/).**
""")

# --- Load Model and Feature List ---
MODEL_PATH = os.path.join("models", "income_rf.pkl")
FEATURES_PATH = os.path.join("models", "model_features.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(FEATURES_PATH, "rb") as f:
    model_features = pickle.load(f)

# --- Input Fields ---
st.header("Enter Individual's Details")

age = st.number_input(
    "Age",
    min_value=17,
    max_value=90,
    value=35,
    help="Person's age in years."
)

education_num = st.number_input(
    "Years of Education",
    min_value=1,
    max_value=16,
    value=13,
    help="Total years of formal education completed (e.g., 13 for high school graduate, 16 for master's degree)."
)

education_options = [
    "Bachelors", "Masters", "HS-grad", "Some-college", "Doctorate",
    "Assoc-acdm", "Assoc-voc", "Prof-school", "11th", "Other"
]
education = st.selectbox(
    "Education Attained",
    education_options,
    index=0,
    help="Highest level of formal education completed."
)

marital_status_options = [
     "Never-married", "Married-civ-spouse", "Divorced",
     "Separated", "Widowed", "Married-spouse-absent"
 ]
 marital_status = st.selectbox(
     "Marital Status",
     marital_status_options,
     index=0,
     help="Current marital status."
 )

occupation_options = [
    "Exec-managerial", "Prof-specialty", "Craft-repair", "Sales",
    "Adm-clerical", "Other-service", "Handlers-cleaners", "Tech-support",
    "Machine-op-inspct", "Farming-fishing", "Transport-moving",
    "Priv-house-serv", "Protective-serv", "Armed-forces"
]
occupation = st.selectbox(
    "Occupation",
    occupation_options,
    index=0,
    help="Primary occupation or job type."
)

workclass_options = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]
workclass = st.selectbox(
    "Workclass",
    workclass_options,
    index=0,
    help="Employment sector or classification."
)

race_options = [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
]
race = st.selectbox(
    "Race",
    race_options,
    index=0,
    help="Racial group as recorded in the US Census."
)

sex = st.radio(
    "Gender",
    ("Male", "Female"),
    help="Biological sex of the individual."
)

hours_per_week = st.number_input(
    "Hours Worked per Week",
    min_value=1,
    max_value=99,
    value=40,
    help="Average hours worked per week."
)

capital_gain = st.number_input(
    "Capital Gain (USD)",
    min_value=0,
    max_value=100000,
    value=0,
    help="Total capital gains (unusual, but can be nonzero for high earners)."
)

# --- One-Hot Encoding Helper ---
def one_hot_encode_input(input_dict, feature_list):
    encoded = {col: 0 for col in feature_list}
    # Numeric features
    for key in ['age', 'education_num', 'hours_per_week', 'capital_gain']:
        if key in input_dict:
            encoded[key] = input_dict[key]
    # Gender
    encoded['sex_male'] = 1 if input_dict['sex'] == 'Male' else 0
    # One-hot categorical
    if f"education_{input_dict['education'].lower()}" in encoded:
        encoded[f"education_{input_dict['education'].lower()}"] = 1
    if f"marital_status_{input_dict['marital_status'].lower()}" in encoded:
        encoded[f"marital_status_{input_dict['marital_status'].lower()}"] = 1
    if f"occupation_{input_dict['occupation'].lower()}" in encoded:
        encoded[f"occupation_{input_dict['occupation'].lower()}"] = 1
    if f"workclass_{input_dict['workclass'].lower()}" in encoded:
        encoded[f"workclass_{input_dict['workclass'].lower()}"] = 1
    # Race mapping for feature name
    race_map = {
        "White": "white", "Black": "black", "Asian-Pac-Islander": "asian-pac-islander",
        "Amer-Indian-Eskimo": "amer-indian-eskimo", "Other": "other"
    }
    race_key = f"race_{race_map[input_dict['race']]}"
    if race_key in encoded:
        encoded[race_key] = 1
    return pd.DataFrame([encoded])

# --- Prepare Input for Model ---
user_input = {
    "age": age,
    "education_num": education_num,
    "education": education,
    "marital_status": marital_status,
    "occupation": occupation,
    "workclass": workclass,
    "race": race,
    "sex": sex,
    "hours_per_week": hours_per_week,
    "capital_gain": capital_gain,
}
input_df = one_hot_encode_input(user_input, model_features)

# --- Prediction ---
predict_btn = st.button("Predict Income")

if predict_btn:
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    label = ">50K" if prediction == 1 else "<=50K"

    if label == ">50K":
        st.markdown(
            f"<h2 style='color:green'>Predicted Income: >$50K (Probability: {prob:.2%})</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h2 style='color:red'>Predicted Income: ≤$50K (Probability: {1-prob:.2%})</h2>",
            unsafe_allow_html=True
        )

    st.caption("Model input features (one-hot encoded):")
    st.dataframe(input_df)

# --- Footer with Links ---
st.markdown("""
---
**Author:** [Dhairya Negandhi](https://www.linkedin.com/in/dhairya-negandhi-933b381bb/)  
:blue_book: [GitHub Repository](https://github.com/DhairyaBN)
""")

