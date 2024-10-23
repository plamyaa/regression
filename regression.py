from abc import ABC, abstractmethod
import numpy as np

class Regression(ABC):
    """
    Abstraction class for regression classes
    """
    def __init__(self, learning_rate: float, max_iter: float, l1: float,
                 l2: float, tolerance: float, verbose: bool):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1 = l1
        self.l2 = l2
        self.tolerance = tolerance
        self.verbose = verbose

    @abstractmethod
    def activation(self, z: np.ndarray) -> np.ndarray:
        """
        Abstract method for applying the activation function (linear or
        sigmoid).
        """
        pass

    @abstractmethod
    def loss(self, y_true: np.ndarray) -> np.ndarray:
        """
        Abstract method for calculating the loss (MSE or log-loss).
        """
        pass

    @abstractmethod
    def threshold(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Abstract method for thresholding the output (used only in
        classification).
        """
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model to the training data.
        """
        n_samples, n_features = np.shape(X)
        X = np.column_stack([np.ones((n_samples)), X])
        self.weights = np.zeros(n_features + 1)

        for step in range(self.max_iter):
            cur_weight = self.weights
            z = X.dot(self.weights)
            f = self.activation(z)
            err = f - y
            grad = X.T.dot(err) / n_samples

            if self.l1 > 0:
                grad += self.l1 * np.sign(self.weights)
            if self.l2 > 0:
                grad += 2 * self.l2 * self.weights

            self.weights = self.weights - self.learning_rate * grad

            if self.verbose:
                loss = self.loss(y, f)
                print(f"Iteration {step + 1}/{self.max_iter}, Loss: {loss:.4f}")

            if np.linalg.norm(cur_weight - self.weights) <= self.tolerance:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Common prediction logic
        """
        n_samples = X.shape[0]
        X = np.column_stack([np.ones((n_samples)), X])
        probabilities = self.activation(X.dot(self.weights))
        return self.threshold(probabilities)
