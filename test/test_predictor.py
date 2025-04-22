from src.predictor import predict_income

def test_sample_person():
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
    result = predict_income(sample)
    assert result in ['>50K', '<=50K']