from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.mlflow_model import RetailGeniusModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "outputs" / "mlflow"

MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_transformed.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load transformed training data and training labels."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")
    return X_train, y_train


def create_model() -> XGBClassifier:
    """Create the XGBoost churn classification model."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def save_model(model: XGBClassifier) -> None:
    """Save the trained model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def main() -> None:
    """Train the churn model and track the experiment with MLflow."""
    print("Loading training data...")
    X_train, y_train = load_training_data()

    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment("RetailGenius-Churn-Prediction")

    with mlflow.start_run() as run:
        print("\nMLflow run started.")
        print(f"Run ID: {run.info.run_id}")

        print("\nCreating XGBoost model...")
        model = create_model()

        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        print("Training model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)

        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        mlflow.log_metrics(
            {
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
            }
        )

        # Save the model and keep the preprocessing artifact available
        # for the MLflow PyFunc wrapper.
        save_model(model)

        print("\nLogging native XGBoost model...")
        mlflow.xgboost.log_model(
            model,
            name="churn_model",
        )

        # Register the native XGBoost model.
        native_model_uri = f"runs:/{run.info.run_id}/churn_model"

        mlflow.register_model(
            model_uri=native_model_uri,
            name="RetailGeniusChurnModel",
        )

        # Create an MLflow PyFunc model that accepts raw customer data.
        print("\nLogging raw-input MLflow model...")

        mlflow.pyfunc.log_model(
            name="retailgenius_raw_input_model",
            python_model=RetailGeniusModel(),
            artifacts={
                "preprocessor": str(PREPROCESSOR_PATH),
                "model": str(MODEL_PATH),
            },
            code_paths=[str(PROJECT_ROOT / "src")],
            pip_requirements=[
                "mlflow==3.16.0",
                "joblib==1.6.0",
                "pandas==2.2.3",
                "scikit-learn==1.9.1",
                "xgboost==3.4.1",
            ],
        )

        raw_model_uri = f"runs:/{run.info.run_id}/retailgenius_raw_input_model"

        mlflow.register_model(
            model_uri=raw_model_uri,
            name="RetailGeniusChurnRawInputModel",
        )

        print("\nTraining completed successfully.")
        print(f"Train Accuracy:  {accuracy:.4f}")
        print(f"Train Precision: {precision:.4f}")
        print(f"Train Recall:    {recall:.4f}")
        print(f"Train F1:        {f1:.4f}")

        print("\nRegistered models:")
        print("1. RetailGeniusChurnModel")
        print("2. RetailGeniusChurnRawInputModel")

    print("\nMLflow tracking completed successfully.")


if __name__ == "__main__":
    main()
