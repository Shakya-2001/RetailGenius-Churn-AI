from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"


def test_dataset_exists():
    """Verify that the raw dataset is available."""
    assert RAW_DATA_PATH.exists()


def test_dataset_has_expected_structure():
    """Verify the dataset contains the expected sheet and columns."""
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="E Comm")

    expected_columns = {
        "CustomerID",
        "Churn",
        "Tenure",
        "PreferredLoginDevice",
        "CityTier",
        "WarehouseToHome",
        "PreferredPaymentMode",
        "Gender",
        "HourSpendOnApp",
        "NumberOfDeviceRegistered",
        "PreferedOrderCat",
        "SatisfactionScore",
        "MaritalStatus",
        "NumberOfAddress",
        "Complain",
        "OrderAmountHikeFromlastYear",
        "CouponUsed",
        "OrderCount",
        "DaySinceLastOrder",
        "CashbackAmount",
    }

    assert set(df.columns) == expected_columns
    assert len(df) == 5630


def test_churn_target_is_binary():
    """Verify that the churn target contains only 0 and 1."""
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="E Comm")

    assert set(df["Churn"].dropna().unique()).issubset({0, 1})
