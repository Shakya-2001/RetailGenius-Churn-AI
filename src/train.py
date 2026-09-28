from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

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
    """Train and save the churn model."""

    print("Loading training data...")

    X_train, y_train = load_training_data()

    print(f"Training features: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")

    print("\nCreating XGBoost model...")

    model = train_model(
        X_train,
        y_train,
    )

    print("Model training completed.")

    save_model(model)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()