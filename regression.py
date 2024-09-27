from abc import ABC, abstractmethod
import numpy as np

class Regression(ABC):
    def __init__(self, learning_rate=0.01, max_iter=1000, l1_ratio=0.0, l2_ratio=0.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _apply_regularization(self, grad):
        if self.l1_ratio > 0:
            grad += self.l1_ratio * np.sign(self.weights)
        if self.l2_ratio > 0:
            grad += 2 * self.l2_ratio * self.weights
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mean_aboslute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
