import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegression:
    def __init__(self, data, labels, poly=1, include_bias=True) -> None:
        self.poly = poly
        self.include_bias = include_bias

        # 更新data
        self.data = PolynomialFeatures(degree=self.poly, include_bias=self.include_bias).fit_transform(data)
        self.labels = labels.reshape(-1, 1)

        # 更新theta
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def fit(self):
        self.theta = np.linalg.inv(self.data.T.dot(self.data)).dot(self.data.T).dot(self.labels)
        return self.theta

    def train(self, alpha, num_iterations=500):
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        cost_history = []
        for _ in range(num_iterations):
            cost = self.gradient_step(alpha)
            cost_history.append(cost)
        return cost_history

    def gradient_step(self, alpha):
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
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
