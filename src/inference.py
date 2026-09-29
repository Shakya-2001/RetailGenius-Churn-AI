from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def load_artifacts():
    """Load the preprocessing pipeline and trained model."""

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    return preprocessor, model


def predict_churn(
    customer_data: pd.DataFrame,
    preprocessor,
    model,
) -> tuple[int, float]:
    """Predict churn for a customer."""

    transformed_data = preprocessor.transform(customer_data)

    prediction = int(model.predict(transformed_data)[0])

    probability = float(model.predict_proba(transformed_data)[0][1])

    return prediction, probability


def main() -> None:
    """Run a sample churn prediction."""

    customer = pd.DataFrame(
        [
            {
                "Tenure": 4,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 1,
                "WarehouseToHome": 10,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "HourSpendOnApp": 3,
                "NumberOfDeviceRegistered": 4,
                "PreferedOrderCat": "Laptop & Accessory",
                "SatisfactionScore": 3,
                "MaritalStatus": "Single",
                "NumberOfAddress": 3,
                "Complain": 1,
                "OrderAmountHikeFromlastYear": 15,
                "CouponUsed": 2,
                "OrderCount": 3,
                "DaySinceLastOrder": 10,
                "CashbackAmount": 150,
            }
        ]
    )

    print("Loading model artifacts...")

    preprocessor, model = load_artifacts()

    print("Artifacts loaded successfully.")

    prediction, probability = predict_churn(
        customer,
        preprocessor,
        model,
    )

    print("\n==============================")
    print("CHURN PREDICTION")
    print("==============================")

    print(f"Prediction: {prediction}")
    print(f"Churn probability: {probability:.4f}")

    if prediction == 1:
        print("Result: Customer is predicted to CHURN.")
    else:
        print("Result: Customer is predicted to NOT CHURN.")


if __name__ == "__main__":
    main()
