import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest

class FeatureSelectionWithANOVA:
    def __init__(self, k=7):
        self.k = k
        self.selected_features = None
        self.feature_scores = None

    def fit(self, X_train, y_train):
        # Perform ANOVA test for feature selection
        fs = SelectKBest(score_func=f_regression, k=self.k)
        fit = fs.fit(X_train, y_train)

        # Get the selected features and their scores
        selected_indices = fit.get_support(indices=True)
        self.selected_features = X_train.columns[selected_indices]
        self.feature_scores = fit.scores_

    def get_selected_features(self):
        return self.selected_features

    def get_feature_scores(self):
        return self.feature_scores
