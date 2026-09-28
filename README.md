# RetailGenius — AI Customer Churn Prediction

RetailGenius is an AI-based e-commerce customer churn prediction project. The system uses customer behavioral and transactional data to predict churn and applies Explainable AI (XAI) using SHAP to understand model predictions.

## Objectives

- Prepare and preprocess customer churn data.
- Build a machine learning model for churn prediction.
- Evaluate the model using standard classification metrics.
- Track experiments and models using MLflow.
- Serve the trained model locally.
- Explain predictions using SHAP.

## Project Structure

```text
RetailGenius-Churn-AI/
│
├── data/
│   ├── raw/
│   │   └── E Commerce Dataset.xlsx
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   ├── mlflow/
│   └── xai/
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   │
│   └── xai/
│       ├── __init__.py
│       └── shap_analysis.py
│
├── tests/
│
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── requirements.txt
├── requirements-xai.txt
├── uv.lock
└── README.md

## Dataset

The project uses the `E Comm` sheet from the E-Commerce Customer Churn dataset.

- Records: 5,630
- Features: 19 input columns after removing `CustomerID`
- Target: `Churn`

Target distribution:

- No Churn: 4,682 (83.16%)
- Churn: 948 (16.84%)

The data is split using stratified train/test splitting with `random_state=42`.

## Machine Learning

The preprocessing pipeline includes:

- Median imputation and scaling for numerical features.
- Most-frequent imputation and one-hot encoding for categorical features.

After preprocessing, the dataset contains 34 features.

The model used is XGBoost.

Current configuration:

- `n_estimators=200`
- `max_depth=5`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`

## Results

Current test-set performance:

| Metric | Score |
|---|---:|
| Accuracy | 94.32% |
| Precision | 88.89% |
| Recall | 75.79% |
| F1-score | 81.82% |
| ROC-AUC | 98.11% |

Confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 918 | 18 |
| Actual 1 | 46 | 144 |

## MLflow

MLflow is used for experiment tracking and model management.

Experiment:

`RetailGenius-Churn-Prediction`

Registered model:

`RetailGeniusChurnModel`

Start the MLflow UI:

`uv run mlflow ui --backend-store-uri "sqlite:///outputs/mlflow/mlflow.db" --port 5000`

The UI is available at:

`http://127.0.0.1:5000`

The registered model can also be served locally on port `5001`.

## Explainable AI

SHAP is used to explain the XGBoost model.

The analysis is implemented in:

`src/xai/shap_analysis.py`

Generated explanations include:

- Mean SHAP feature importance
- Beeswarm plot
- Class-specific summary plots
- Waterfall plot
- Force plot
- Dependence plot

Outputs are stored in:

`outputs/xai/`

The current most influential feature by mean absolute SHAP value is:

`numerical__Tenure`

## Running the Project

Install dependencies:

`uv sync`

Run the pipeline:

`uv run python src/data_preparation.py`

`uv run python src/feature_engineering.py`

`uv run python src/train.py`

`uv run python src/evaluate.py`

`uv run python src/inference.py`

Run SHAP analysis:

`.xai-venv\Scripts\python.exe src\xai\shap_analysis.py`

## Technologies

- Python 3.12
- Pandas
- Scikit-learn
- XGBoost
- MLflow
- SHAP
- Matplotlib
- Joblib
- uv
- Git