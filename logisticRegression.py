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

    def activation(self, z: np.ndarray) -> np.ndarray:
        """
        Activation method to apply the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def threshold(self, probabilities: np.ndarray) -> np.ndarray:
        """
        No thresholding for linear regression, return the raw predictions.
        """
        return probabilities

    def loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute the log loss (binary cross-entropy) between true labels
        and predicted probabilities.
        """
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return (-np.mean(y_true * np.log(y_pred_proba) 
                         + (1 - y_true) * np.log(1 - y_pred_proba)))