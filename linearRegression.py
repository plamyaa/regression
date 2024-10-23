from regression import Regression
import numpy as np

class LinearRegression(Regression):
    """
    A class for linear regression with support for gradient descent,
    analytical solutions and L1/L2 regularization.
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
        self.bias = None

    def fit_analytical(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using the analytical solution.
        """
        X = np.insert(X, 0, 1, axis=1) # Add bias 
        XT_X_inv = np.linalg.inv(X.T @ X)  
        self.weights = np.linalg.multi_dot([XT_X_inv, X.T, y])

    def activation(self, z: np.ndarray) -> np.ndarray:
        """
        Linear activation: return the input as is.
        """
        return z

    def threshold(self, probabilities: np.ndarray) -> np.ndarray:
        """
        No thresholding for linear regression, return the raw predictions.
        """
        return probabilities

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error loss for linear regression.
        """
        return np.mean((y_true - y_pred) ** 2)
