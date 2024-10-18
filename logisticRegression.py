from regression import Regression
import numpy as np

class LogisticRegression(Regression):
    """
    A class for logistic regression with support L1/L2 regularization.
    """
    def __init__(self, learning_rate: float=0.01, max_iter: int=1000,
                 l1: float=0.0, l2: float=0.0, tolerance=1e-4,
                 verbose: bool=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2
        self.tolerance = tolerance
        self.verbose = verbose
        self.weights = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Private method to apply the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression model to the training data.
        """
        n_samples, n_features = np.shape(X)
        X = np.column_stack([np.ones((n_samples)), X])
        self.weights = np.zeros(n_features + 1)

        for step in range(self.max_iter):
            cur_weight = self.weights
            z = X.dot(self.weights)
            f = self._sigmoid(z)
            err = f - y
            grad = X.T.dot(err) / n_samples

            if self.l1 > 0:
                grad += self.l1 * np.sign(self.weights)
            if self.l2 > 0:
                grad += 2 * self.l2 * self.weights

            self.weights = self.weights - self.learning_rate * grad

            if self.verbose:
                logloss = self.log_loss(y, self.predict_proba(X))
                mse = self.mean_squared_error(y, f)
                print(f"Iteration {step + 1}/{self.max_iter}, MSE: {mse:.4f}, ",
                      f"Logloss: {logloss:.4f}")

            if np.linalg.norm(cur_weight - self.weights) <= self.tolerance:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for the input data.
        """
        n_samples = X.shape[0]
        X = np.column_stack([np.ones((n_samples)), X])
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of the positive class for the input data.
        """
        return self._sigmoid(X.dot(self.weights))

    def log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute the log loss (binary cross-entropy) between true labels
        and predicted probabilities.
        """
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return (-np.mean(y_true * np.log(y_pred_proba) 
                         + (1 - y_true) * np.log(1 - y_pred_proba)))