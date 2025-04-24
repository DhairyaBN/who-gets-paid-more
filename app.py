import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# --- Load Model and Features ---
MODEL_PATH = os.path.join('models', 'income_rf.pkl')
FEATURES_PATH = os.path.join('models', 'model_features.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    model_features = pickle.load(f)

# --- Streamlit UI ---
st.title("Who Gets Paid More? Income Prediction App")
st.write(
    "Fill in the fields below and predict if a person will earn >$50K or <=$50K based on the trained model."
)

# Simple UI for the most important features
age = st.number_input("Age", min_value=17, max_value=90, value=35)
education_num = st.number_input("Years of Education (education_num)", min_value=1, max_value=16, value=13)
hours_per_week = st.number_input("Hours Worked per Week", min_value=1, max_value=99, value=40)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
sex_male = st.radio("Gender", ['Male', 'Female']) == 'Male'

# Add a few categorical features (one-hot encoded)
education_bachelors = st.checkbox("Bachelor's Degree")
occupation_exec_managerial = st.checkbox("Exec-Managerial Occupation")
marital_status_married_civ_spouse = st.checkbox("Married (Civ-Spouse)")
native_country_us = st.checkbox("Native Country: United States", value=True)
workclass_private = st.checkbox("Workclass: Private", value=True)
race_white = st.checkbox("Race: White", value=True)

# --- Prepare Input Row ---
input_dict = {name: 0 for name in model_features}
input_dict.update({
    'age': age,
    'education_num': education_num,
    'hours_per_week': hours_per_week,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'sex_male': int(sex_male),
    'education_bachelors': int(education_bachelors),
    'occupation_exec-managerial': int(occupation_exec_managerial),
    'marital_status_married-civ-spouse': int(marital_status_married_civ_spouse),
    'native_country_united-states': int(native_country_us),
    'workclass_private': int(workclass_private),
    'race_white': int(race_white),
})

input_df = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: **{label}**")

    st.write("Raw input row sent to model:")
    st.dataframe(input_df)
