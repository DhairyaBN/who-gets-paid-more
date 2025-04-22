# Who Gets Paid More?

Predict whether a person earns >$50K using demographic and employment data.

## 📁 Project Structure

- `notebooks/`: EDA and training (Jupyter notebooks)
- `src/`: Standalone model prediction code
- `models/`: Trained Random Forest model and feature list
- `test/`: Optional unit test

## 🚀 How to Use

```bash
git clone https://github.com/yourusername/who-gets-paid-more.git
cd who-gets-paid-more
```

Then in Python:

```python
from src.predictor import predict_income

sample = {
    'age': 35,
    'education_num': 13,
    'hours_per_week': 40,
    'capital_gain': 0,
    'capital_loss': 0,
    'sex_male': 1,
    'education_bachelors': 1,
    'occupation_exec-managerial': 1,
    'marital_status_married-civ-spouse': 1,
    'native_country_united-states': 1,
    'workclass_private': 1,
    'race_white': 1
}

print(predict_income(sample))
```

## ✅ Requirements

- pandas
- numpy
- scikit-learn
- (optional) pytest for testing