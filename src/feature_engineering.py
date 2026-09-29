from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


NUMERICAL_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def main() -> None:
    """Fit preprocessing on training data and transform train/test data."""

    print("Loading prepared datasets...")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    preprocessor = create_preprocessor()

    print("\nFitting preprocessor on training data...")

    X_train_transformed = preprocessor.fit_transform(X_train)

    print("Transforming test data...")

    X_test_transformed = preprocessor.transform(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    pd.DataFrame(X_train_transformed).to_csv(
        PROCESSED_DATA_DIR / "X_train_transformed.csv",
        index=False,
    )

    pd.DataFrame(X_test_transformed).to_csv(
        PROCESSED_DATA_DIR / "X_test_transformed.csv",
        index=False,
    )

    print("\nFeature engineering completed successfully.")
    print(f"Transformed training shape: {X_train_transformed.shape}")
    print(f"Transformed testing shape: {X_test_transformed.shape}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
