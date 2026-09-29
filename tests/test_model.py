from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_test_transformed.csv"


def test_model_files_exist():
    """Verify that trained model artifacts exist."""
    assert MODEL_PATH.exists()
    assert PREPROCESSOR_PATH.exists()


def test_model_can_be_loaded():
    """Verify that the trained model can be loaded."""
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_prediction_output():
    """Verify that the model produces valid predictions."""
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(TEST_DATA_PATH)

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_model_probability_output():
    """Verify that churn probabilities are between 0 and 1."""
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(TEST_DATA_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(X_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
