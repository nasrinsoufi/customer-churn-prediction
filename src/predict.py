from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"


def load_model():
    """Load the trained churn prediction pipeline."""
    return joblib.load(MODEL_PATH)

def predict_churn(customer_data: dict, model=None):
    """Predict whether a customer is likely to churn."""

    if model is None:
        model = load_model()

    expected_features = list(model.feature_names_in_)

    # Check for missing features
    missing_features = [
        feature for feature in expected_features
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Ignore unexpected features and preserve column order
    customer_df = pd.DataFrame(
        [{feature: customer_data[feature]
          for feature in expected_features}]
    )

    # Predict churn probability
    churn_probability = model.predict_proba(
        customer_df
    )[0, 1]

    threshold = 0.5
    prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": round(
            float(churn_probability), 4
        ),
        "prediction": prediction,
        "prediction_label": (
            "Yes" if prediction == 1 else "No"
        )
    }
