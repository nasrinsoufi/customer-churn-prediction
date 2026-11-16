# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave the company (churn) using customer demographics, subscribed services, contract details, and billing information.

The project covers the complete machine learning workflow, from data exploration and preprocessing to model evaluation, prediction, automated testing, and API development.

## Project Overview

Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to leave can help businesses take proactive retention measures.

This project uses supervised machine learning to predict customer churn as a binary classification problem.

- **Target:** Churn (Yes / No)
- **Dataset:** IBM Telco Customer Churn
- **Samples:** 7,043 customers
- **Features:** 19 input features
- **Model:** Logistic Regression with class weighting

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- FastAPI
- Pydantic
- Pytest
- Joblib

## Machine Learning Workflow

1. Data exploration and analysis
2. Data cleaning and handling missing values
3. Train/test splitting with stratification
4. Numerical feature scaling using StandardScaler
5. Categorical feature encoding using OneHotEncoder
6. Model training using Logistic Regression
7. Handling class imbalance with `class_weight="balanced"`
8. Model evaluation using Precision, Recall, F1-score, and Confusion Matrix
9. Saving the trained pipeline
10. Building a prediction module and REST API

## Model Evaluation

The final model was evaluated on a held-out test set containing 1,409 customers.

| Metric | No Churn (0) | Churn (1) |
|---|---:|---:|
| Precision | 0.90 | 0.50 |
| Recall | 0.72 | 0.78 |
| F1-score | 0.80 | 0.61 |

**Overall Accuracy:** 73.81%

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 747 | 288 |
| Actual Yes | 81 | 293 |

The model identifies 78% of customers who actually churned. This comes with a trade-off: some customers predicted to churn will not actually leave.

The choice of evaluation metric depends on business priorities. For example, a company that wants to identify as many potential churners as possible may prioritize Recall.

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── raw/
├── models/
│   └── churn_pipeline.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── predict.py
├── tests/
│   ├── test_api.py
│   └── test_predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nasrinsoufi/customer-churn-prediction.git
cd customer-churn-prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Dataset and trained model:** The raw dataset and trained model are not included in the repository. Place the dataset in `data/raw/` and generate the trained pipeline at `models/churn_pipeline.joblib` by running the training workflow in the notebook.

## Run the API

From the project root:

```bash
python -m uvicorn src.main:app --reload
```

API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- API endpoint: `POST /predict`

## Example API Request

```json
{
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
```

Example response:

```json
{
  "churn_probability": 0.9126,
  "prediction": 1,
  "prediction_label": "Yes"
}
```

## Run Tests

```bash
python -m pytest -v
```

## Dataset

The project uses the IBM Telco Customer Churn dataset, available on Kaggle:

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telecom-customer-churn)

