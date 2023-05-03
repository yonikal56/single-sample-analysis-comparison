import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


class ROC:
    def __init__(self, plot=True):
        self.__plot = plot
        if plot:
            plt.figure()
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')

    def get_points(self, real, predicted_values):
        fpr = dict()
        tpr = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(real, predicted_values)
        return fpr[1], tpr[1], roc_auc_score(real, predicted_values)

    def add_graph(self, real, predicted_values, label):
        x, y, auc = self.get_points(real, predicted_values)
        if self.__plot:
            plt.plot(x, y, label=f'{label} - {round(auc, 2)}')
        return auc

    def show(self):
        if self.__plot:
            plt.legend()
            plt.show()
