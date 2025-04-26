import pickle
import pandas as pd
import numpy as np
import os

# Load model and feature names
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'income_rf.pkl')
features_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_features.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(features_path, 'rb') as f:
    model_features = pickle.load(f)

def predict_income(user_input: dict) -> str:
    """
    Predict income class based on input features.

    Args:
        user_input (dict): keys are feature names, values are inputs.

    Returns:
        str: '>50K' or '<=50K'
    """
    input_row = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    for k, v in user_input.items():
        if k in input_row.columns:
            input_row.at[0, k] = v
    prediction = model.predict(input_row)[0]
    return '>50K' if prediction == 1 else '<=50K'
