from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_data() -> pd.DataFrame:
    """Load the E-Commerce churn dataset."""
    return pd.read_excel(
        RAW_DATA_PATH,
        sheet_name="E Comm",
    )


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for machine learning."""

    # CustomerID is an identifier, not a predictive feature.
    df = df.drop(columns=["CustomerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def save_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save prepared datasets to disk."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)


def main() -> None:
    """Run the data preparation pipeline."""

    print("Loading dataset...")

    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    X, y = prepare_data(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(4))

    save_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nPrepared datasets saved successfully.")


if __name__ == "__main__":
    main()