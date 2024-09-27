from regression import Regression
import numpy as np

class LinearRegression(Regression):
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
    
    def fit_analytical(self, X, y):
        X = np.insert(X, 0, 1, axis=1) # учесть смещение w0
        XT_X_inv = np.linalg.inv(X.T @ X) # (X^T * X)^-1
        self.weights = np.linalg.multi_dot([XT_X_inv, X.T, y]) # (X^T * X)^-1 * X^T * y

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.weights = np.zeros(n_features)

        for step in range(self.max_iter):
            f = X.dot(self.weights)
            err = f - y
            grad = 2 * X.T.dot(err) / n_samples
            self.weights -= self.learning_rate * grad

    def predict(self, X):
        return X.dot(self.weights)
