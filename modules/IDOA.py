import numpy as np
import math
from modules.DOC import DOC


class IDOA:
    def __init__(self, data):
        self.__data = data
        self.__doc = DOC()

    def calculate_IDOA(self, cohort, sample, draw=False, graphs=None, axs=None):
        doc_points = [self.__doc.get_dissimilarity_overlap_point(sample, co) for co in cohort]
        x = []
        y = []
        for dis, ov in doc_points:
            if ov >= 0.5:
                x.append(ov)
                y.append(dis)
        idoa_value = np.polyfit(x, y, 1)[0]
        if draw:
            graphs.scatter(x, y, axs)
        return idoa_value

    def predict(self, samples, include_values=False):
        results = []
        for sample in samples:
            idoa_values = []
            for model in self.__data['models']:
                idoa_values.append(self.calculate_IDOA(model["cohort"], sample, False))
            if include_values:
                results.append([idoa_values.index(min(idoa_values)), idoa_values])
            else:
                results.append(idoa_values.index(min(idoa_values)))
        return results

    # def calculate_upper_value(self, cohort):
    #     idoa_values = [self.calculate_IDOA(cohort[:i] + cohort[i+1:], cohort[i]) for i in range(len(cohort))]
    #     idoa_values.sort()
    #     bound = round(len(cohort) * 0.95)
    #     return idoa_values[bound - 1]

    def predict_real(self, cohort, samples):
        return [self.calculate_IDOA(cohort, sample, False) for sample in samples]

    def __str__(self):
        return "IDOA"
