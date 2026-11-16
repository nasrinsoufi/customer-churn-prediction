from fastapi.testclient import TestClient

from src.main import app
import src.main as main_module


client = TestClient(app)


def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }


def test_predict_endpoint_returns_prediction(monkeypatch):
    def fake_predict(customer_data):
        return {
            "churn_probability": 0.85,
            "prediction": 1,
            "prediction_label": "Yes"
        }

    monkeypatch.setattr(
        main_module,
        "predict_churn",
        fake_predict
    )

    response = client.post(
        "/predict",
        json=valid_customer()
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["prediction_label"] == "Yes"


def test_predict_endpoint_rejects_invalid_gender():
    customer = valid_customer()
    customer["gender"] = "Unknown"

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_field():
    customer = valid_customer()
    del customer["tenure"]

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422
