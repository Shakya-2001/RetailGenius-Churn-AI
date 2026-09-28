from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "figures"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_transformed.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load transformed test data and test labels."""

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    return X_test, y_test


def load_model():
    """Load the trained churn model."""

    return joblib.load(MODEL_PATH)


def evaluate_model(model, X_test, y_test) -> None:
    """Evaluate the model using classification metrics."""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(confusion_matrix(y_test, y_pred))

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Churn",
                "Churn",
            ],
        )
    )


def main() -> None:
    """Evaluate the trained churn model."""

    print("Loading test data...")

    X_test, y_test = load_test_data()

    print(f"Test features: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    print("\nLoading trained model...")

    model = load_model()

    print("Model loaded successfully.")

    evaluate_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()