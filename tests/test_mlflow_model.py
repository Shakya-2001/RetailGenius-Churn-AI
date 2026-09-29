import joblib
import pandas as pd

from src.mlflow_model import RetailGeniusModel

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def test_mlflow_model_prediction():
    """Test raw customer prediction through the MLflow wrapper."""
    customer = pd.DataFrame(
        [
            {
                "Tenure": 4,
                "PreferredLoginDevice": "Mobile Phone",
                "CityTier": 3,
                "WarehouseToHome": 6,
                "PreferredPaymentMode": "Debit Card",
                "Gender": "Male",
                "HourSpendOnApp": 3,
                "NumberOfDeviceRegistered": 4,
                "PreferedOrderCat": "Laptop & Accessory",
                "SatisfactionScore": 3,
                "MaritalStatus": "Single",
                "NumberOfAddress": 2,
                "Complain": 1,
                "OrderAmountHikeFromlastYear": 11,
                "CouponUsed": 1,
                "OrderCount": 2,
                "DaySinceLastOrder": 5,
                "CashbackAmount": 150,
            }
        ]
    )

    wrapper = RetailGeniusModel()

    wrapper.preprocessor = joblib.load(PREPROCESSOR_PATH)
    wrapper.model = joblib.load(MODEL_PATH)

    result = wrapper.predict(None, customer)

    assert isinstance(result, pd.DataFrame)
    assert "prediction" in result.columns
    assert "churn_probability" in result.columns

    assert len(result) == 1
    assert result["prediction"].iloc[0] in [0, 1]
    assert 0 <= result["churn_probability"].iloc[0] <= 1
