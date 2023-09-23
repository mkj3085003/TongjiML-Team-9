# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess_data(self, df):
        column_names = ["gdp_per_capita", "health", "cpi_score", "freedom", "government_trust", "social_support",
                        "dystopia_residual", "Country", "Year", "continent", "generosity", "family",
                        "happiness_score"]
        df = df.reindex(columns=column_names)

        X = df.drop(["happiness_score", "Country", "continent", "Year", "generosity", "family"], axis=1)
        y = df["happiness_score"]

        scale = MinMaxScaler()
        X_scaled = scale.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test
