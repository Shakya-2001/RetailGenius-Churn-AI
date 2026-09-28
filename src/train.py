from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "outputs" / "mlflow"

MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_transformed.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"


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


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train the churn prediction model."""

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    return model


def save_model(model: XGBClassifier) -> None:
    """Save the trained model."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )


def main() -> None:
    """Train the churn model and track the experiment with MLflow."""

    print("Loading training data...")

    X_train, y_train = load_training_data()

    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")

    # Configure local MLflow tracking.
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    )

    # Create or reuse the MLflow experiment.
    mlflow.set_experiment("RetailGenius-Churn-Prediction")

    with mlflow.start_run() as run:
        print("\nMLflow run started.")
        print(f"Run ID: {run.info.run_id}")

        print("\nCreating XGBoost model...")

        model = create_model()

        # Log model parameters.
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

        model.fit(
            X_train,
            y_train,
        )

        # Calculate training metrics.
        y_pred = model.predict(X_train)

        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        # Log metrics.
        mlflow.log_metrics(
            {
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_f1": f1,
            }
        )

        # Log the XGBoost model to MLflow.
        mlflow.xgboost.log_model(
            model,
            name="churn_model",
        )

        # Register the model in the MLflow Model Registry.
        model_uri = f"runs:/{run.info.run_id}/churn_model"

        mlflow.register_model(
            model_uri=model_uri,
            name="RetailGeniusChurnModel",
        )

        # Also save our normal joblib model.
        save_model(model)

        print("\nModel training completed.")

        print("\nTraining metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"MLflow run ID: {run.info.run_id}")

    print("\nMLflow tracking completed successfully.")


if __name__ == "__main__":
    main()