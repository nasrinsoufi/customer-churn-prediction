from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_churn
from typing import Literal

class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]

    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]

    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]

    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer_data: CustomerData):

    try:
        result = predict_churn(customer_data.model_dump())
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )