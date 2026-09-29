import joblib
import mlflow.pyfunc
import pandas as pd


class RetailGeniusModel(mlflow.pyfunc.PythonModel):
    """MLflow model combining preprocessing and churn prediction."""

    def load_context(self, context):
        """Load the preprocessing pipeline and trained model."""
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Transform raw customer data and return churn predictions."""
        transformed_data = self.preprocessor.transform(model_input)

        predictions = self.model.predict(transformed_data)
        probabilities = self.model.predict_proba(transformed_data)[:, 1]

        return pd.DataFrame(
            {
                "prediction": predictions.astype(int),
                "churn_probability": probabilities,
            }
        )
