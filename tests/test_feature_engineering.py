from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_train_transformed.csv"
X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test_transformed.csv"


def test_transformed_datasets_exist():
    """Verify transformed training and testing datasets exist."""
    assert X_TRAIN_PATH.exists()
    assert X_TEST_PATH.exists()


def test_transformed_dataset_shape():
    """Verify transformed datasets have the expected dimensions."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    assert X_train.shape == (4504, 34)
    assert X_test.shape == (1126, 34)


def test_transformed_dataset_contains_no_missing_values():
    """Verify feature engineering produced complete datasets."""
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
