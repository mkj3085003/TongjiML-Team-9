import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegression:
    def __init__(self, data, labels, poly=1, include_bias=True, regularization=False) -> None:
        self.poly = poly
        self.include_bias = include_bias
        self.regularization = regularization

        # 更新data
        self.data = PolynomialFeatures(degree=self.poly, include_bias=self.include_bias).fit_transform(data)
        self.labels = labels.reshape(-1, 1)

        # 更新theta
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def fit(self, lamda=None):
        if self.regularization:
            self.theta = np.linalg.inv(self.data.T.dot(self.data) + lamda * np.eye(self.data.shape[1])).dot(
                self.data.T).dot(self.labels)
        else:
            self.theta = np.linalg.inv(self.data.T.dot(self.data)).dot(self.data.T).dot(self.labels)
        return self.theta

    def train(self, alpha, num_iterations=500, lamda=None):
        cost_history = self.gradient_descent(alpha, num_iterations, lamda)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations, lamda=None):
        cost_history = []
        for _ in range(num_iterations):
            cost = self.gradient_step(alpha, lamda)
            cost_history.append(cost)
        return cost_history

    def gradient_step(self, alpha, lamda=None):
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        if self.regularization:
            theta[1:] = (theta[1:] - alpha * (1 / num_examples) * (np.dot(delta.T, self.data[:, 1:])).T
                         - alpha * lamda * theta[1:])
            theta[0] = theta[0] - alpha * (1 / num_examples) * np.sum(delta)
        else:
            theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T

        self.theta = theta

        cost = (1 / 2) * np.dot(delta.T, delta)
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data, theta)
        return predictions

    def predict(self, data):
        data_processed = PolynomialFeatures(degree=self.poly, include_bias=self.include_bias).fit_transform(data)
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions
