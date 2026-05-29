from sklearn.linear_model import LogisticRegression
import numpy as np

class FraudModel:
    """A classic machine learning classification pipeline predicting transaction safety."""
    def __init__(self):
        self.model = LogisticRegression()
        self._train_mock_model()

    def _train_mock_model(self):
        # Features matrix: [Amount, FailureCount]
        X = np.array([[10.0, 0], [15.0, 0], [5000.0, 4], [3500.0, 3], [20.0, 0], [4500.0, 5]])
        # Labels vector: 0 = Clear, 1 = Fraud flag
        y = np.array([0, 0, 1, 1, 0, 1])
        self.model.fit(X, y)

    def predict_fraud_risk(self, amount: float, login_failures: int) -> bool:
        """Evaluates live input metrics against the trained classification boundary."""
        prediction = self.model.predict([[amount, login_failures]])
        return bool(prediction[0])
