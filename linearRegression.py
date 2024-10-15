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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model using gradient descent.
        """
        n_samples, n_features = X.shape
        X = np.column_stack([np.ones((n_samples)), X])
        self.weights = np.zeros(n_features + 1)

        for step in range(self.max_iter):
            cur_weight = self.weights
            f = X.dot(self.weights)
            err = f - y
            grad = 2 * X.T.dot(err) / n_samples

            if self.l1 > 0:
                grad += self.l1 * np.sign(self.weights)
            if self.l2 > 0:
                grad += 2 * self.l2 * self.weights

            self.weights = self.weights - self.learning_rate * grad # Antigradient

            if self.verbose:
                mse = self.mean_squared_error(y, f)
                print(f"Iteration {step + 1}/{self.max_iter}, MSE: {mse:.4f}")

            if np.linalg.norm(cur_weight - self.weights) <= self.tolerance:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        """
        n_samples = X.shape[0]
        X = np.column_stack([np.ones((n_samples)), X])
        return X.dot(self.weights)  # \hat{y} = X * w
