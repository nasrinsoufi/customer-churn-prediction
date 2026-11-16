import pytest
import pandas as pd
from src.predict import predict_churn
import numpy as np

class FakeModel:
    feature_names_in_ = ["gender", "tenure", "MonthlyCharges"]

    def predict_proba(self, data):
     return np.array([[0.1, 0.9]])


def test_predict_churn_returns_expected_output():
    customer = {
        "gender": "Female",
        "tenure": 2,
        "MonthlyCharges": 85.5
    }

    result = predict_churn(customer, model=FakeModel())

    assert result["churn_probability"] == 0.9
    assert result["prediction"] == 1
    assert result["prediction_label"] == "Yes"

def test_predict_churn_raises_error_for_missing_features():
    customer = {
        "gender": "Female",
        "MonthlyCharges": 85.5
    }

    with pytest.raises(ValueError, match="Missing features"):
        predict_churn(customer, model=FakeModel())


def test_predict_churn_ignores_unexpected_features():
    customer = {
        "gender": "Female",
        "tenure": 2,
        "MonthlyCharges": 85.5,
        "customerID": "TEST-001"
    }

    result = predict_churn(customer, model=FakeModel())

    assert result["prediction"] == 1