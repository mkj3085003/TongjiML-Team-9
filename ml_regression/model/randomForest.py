from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class RandomForestRegressor:
    def __init__(self, data, labels, n_estimators=100, random_state=42):
        self.data = data
        self.labels = labels.reshape(-1, 1)
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.forest = []
        for _ in range(self.n_estimators):
            self.forest.append(DecisionTreeRegressor(random_state=self.random_state))

    def train(self, size=0.6):
        for i in range(self.n_estimators):
            X_reserve, X_abandon, Y_reserve, Y_abandon = train_test_split(self.data, self.labels, test_size=size,
                                                                          random_state=i * 5)
            self.forest[i].fit(X_reserve, Y_reserve)

    def predict(self, data):
        result = 0
        for i in range(self.n_estimators):
            result += self.forest[i].predict(data)
        result /= self.n_estimators
        return result
