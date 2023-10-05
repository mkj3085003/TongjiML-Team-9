import numpy as np
from copy import deepcopy


def objective_function(y, y_hat):
    """
    均方误差损失函数
    :param y: [n_samples,]
    :param y_hat:  [n_samples,]
    :return:  [n_samples,]
    """
    return 0.5 * (y - y_hat) ** 2


def negative_gradient(y, y_hat):
    """
    J 关于 y_hat 的负梯度
    由于后续需要用每个样本的梯度来更新预测结果
    所以这里并不用计算整体的平均梯度
    -(-(y- y_hat))
    :param y:
    :param y_hat:
    :return:
    """
    return (y - y_hat)


class MyGradientBoostRegression(object):
    """
    Gradient Boosting for Regression
    Parameters:
        learning_rate: 学习率
        n_estimators: boosting 次数
        base_estimator: 预测梯度时使用到的回归模型
    """

    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 base_estimator=None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.loss_ = []

    def fit(self, X, y):
        """

        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self.raw_predictions = np.zeros(X.shape[0])  # [n_samples,]
        self._fit_stages(X, y, self.raw_predictions)

    def _fit_stages(self, X, y, raw_predictions):
        for i in range(self.n_estimators):
            grad = negative_gradient(y, raw_predictions)  # 计算负梯度
            model = deepcopy(self.base_estimator)
            model.fit(X, grad)  # 这里其实是用于拟合梯度，因为在预测时无法计算得到真实梯度
            grad = model.predict(X)  # 当然，这里的grad也可以直接使用上面的真实值
            raw_predictions += self.learning_rate * grad  # 梯度下降更新预测结果
            self.loss_.append(np.sum(objective_function(y, raw_predictions)))
            self.estimators_.append(model)  # 保存每个模型

    def predict(self, X):
        """
        模型预测
        :param X: [n_samples, n_features]
        :return: [n_samples,]
        """
        raw_predictions = np.zeros(X.shape[0])  # [n_samples,]
        for model in self.estimators_:
            grad = model.predict(X)  # 预测每个样本在当前boost序列中对应的梯度值
            raw_predictions += self.learning_rate * grad  # 梯度下降更新预测结果
        return raw_predictions