import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_test_transformed.csv"
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "processed" / "feature_names.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai"


def load_model_and_data():
    """Load model, transformed test data, and feature names."""
    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(TEST_DATA_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    X_test.columns = feature_names

    return model, X_test


def main():
    """Generate SHAP explanations for the churn model."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test = load_model_and_data()

    print("Model loaded successfully.")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {len(X_test.columns)}")

    # ---------------------------------------------------------------
    # Create TreeExplainer
    # ---------------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    print("TreeExplainer created successfully.")

    # ---------------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------------

    shap_values = explainer(X_test)

    print("SHAP values calculated successfully.")
    print(f"SHAP values shape: {shap_values.values.shape}")

    # ---------------------------------------------------------------
    # 1. Mean SHAP feature importance
    # ---------------------------------------------------------------

    mean_abs_shap = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": (abs(shap_values.values).mean(axis=0)),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    mean_shap_path = OUTPUT_DIR / "mean_shap.csv"
    mean_abs_shap.to_csv(mean_shap_path, index=False)

    print(f"Saved: {mean_shap_path}")

    # ---------------------------------------------------------------
    # 2. Beeswarm plot
    # ---------------------------------------------------------------

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    beeswarm_path = OUTPUT_DIR / "beeswarm.png"
    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {beeswarm_path}")

    # ---------------------------------------------------------------
    # 3. Waterfall plot for customer 0
    # ---------------------------------------------------------------

    customer_index = 0

    shap.plots.waterfall(
        shap_values[customer_index],
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    waterfall_path = OUTPUT_DIR / "waterfall_customer_0.png"

    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {waterfall_path}")

    # ---------------------------------------------------------------
    # 2b. Class-specific SHAP summary plots
    # ---------------------------------------------------------------

    # Class 0: features pushing predictions toward non-churn
    class_0_values = shap_values.values.copy()
    class_0_values[class_0_values > 0] = 0

    class_0_explanation = shap.Explanation(
        values=class_0_values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )

    shap.plots.beeswarm(
        class_0_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    class_0_path = OUTPUT_DIR / "summary_class_0.png"

    plt.savefig(
        class_0_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {class_0_path}")

    # Class 1: features pushing predictions toward churn
    class_1_values = shap_values.values.copy()
    class_1_values[class_1_values < 0] = 0

    class_1_explanation = shap.Explanation(
        values=class_1_values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )

    shap.plots.beeswarm(
        class_1_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    class_1_path = OUTPUT_DIR / "summary_class_1.png"

    plt.savefig(
        class_1_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {class_1_path}")

    # ---------------------------------------------------------------
    # 2c. Mean SHAP feature importance plot
    # ---------------------------------------------------------------

    top_mean_shap = mean_abs_shap.head(15).sort_values("mean_abs_shap")

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_mean_shap["feature"],
        top_mean_shap["mean_abs_shap"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Mean SHAP Feature Importance")

    plt.tight_layout()

    mean_shap_plot_path = OUTPUT_DIR / "mean_shap.png"

    plt.savefig(
        mean_shap_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {mean_shap_plot_path}")

    # ---------------------------------------------------------------
    # 4. Force plot for customer 0
    # ---------------------------------------------------------------

    shap.plots.force(
        shap_values[customer_index],
        matplotlib=True,
        show=False,
    )

    plt.tight_layout()

    force_path = OUTPUT_DIR / "force_customer_0.png"

    plt.savefig(
        force_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {force_path}")

    # ---------------------------------------------------------------
    # 5. Dependence / scatter plot
    # ---------------------------------------------------------------

    most_important_feature = mean_abs_shap.iloc[0]["feature"]

    shap.plots.scatter(
        shap_values[:, most_important_feature],
        color=shap_values,
        show=False,
    )

    plt.tight_layout()

    dependence_path = OUTPUT_DIR / "dependence.png"

    plt.savefig(
        dependence_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        "Most important feature:",
        most_important_feature,
    )
    print(f"Saved: {dependence_path}")

    # ---------------------------------------------------------------
    # 6. Print top features
    # ---------------------------------------------------------------

    print("\nTop 10 features by mean absolute SHAP value:")

    print(mean_abs_shap.head(10).to_string(index=False))

    print("\nXAI analysis completed successfully.")


if __name__ == "__main__":
    main()
