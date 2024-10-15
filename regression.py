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
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def mean_squared_error(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        """
        return np.mean((y - y_pred) ** 2)
    
    def root_mean_squared_error(self, y: np.ndarray,
                                y_pred: np.ndarray) -> float:
        """
         Calculate the Root Mean Squared Error (RMSE).
        """
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    def mean_absolute_error(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Error (MAE).
        """
        return np.mean(np.abs(y - y_pred))

    def mean_aboslute_percentage_error(self, y: np.ndarray,
                                       y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE)
        """
        return np.mean(np.abs((y - y_pred) / y)) * 100

    def r_squared(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared (coefficient of determination)
        """
        redisual_variance = np.sum((y - y_pred) ** 2)
        total_variance = np.sum((y - np.mean(y_pred)) ** 2)
        return (1 - (redisual_variance / total_variance) 
                if total_variance != 0 
                else 0)