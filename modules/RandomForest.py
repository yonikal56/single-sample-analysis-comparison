import numpy as np
import pandas
from modules.GLV import GLV
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, data):
        self.__model = None
        self.__data = data
        self.train()

    def train(self):
        X = []
        y = []
        # create training data arrays
        cohorts = len(self.__data['models'])
        for n in range(1, cohorts + 1):
            m = len(self.__data['models'][n - 1]["cohort"])
            X.extend(self.__data['models'][n - 1]["cohort"])
            y += [[0] * (n - 1) + [1] + [0] * (cohorts - n)] * m

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.__model = model


    def predict(self, samples):
        # prediction for all samples
        all_predicts = self.__model.predict(samples).tolist()
        # return the class with the maximum prediction value
        return [predict.index(max(predict)) for predict in all_predicts]

    def predict_real(self, cohort, samples):
        # prediction for all samples
        all_predicts = self.__model.predict(samples).tolist()
        # return prediction for class 0
        return [predict[0] for predict in all_predicts]

    def __str__(self):
        return "Random Forest"
