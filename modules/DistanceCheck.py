import numpy as np
from scipy.spatial import distance


class DistanceCheck:
    methods = {'Bray-Curtis': (distance.braycurtis, 0.3),
               'Euclidean': (distance.euclidean, 2)}

    def __str__(self):
        return f'{list(self.methods.keys())[self.__method]} Dissimilarity'

    def __init__(self, data, method=0):
        self.__data = data
        self.__method = method

    @staticmethod
    def calculate_in_group_distance(cohort, method=0):
        distances = []
        for i in range(len(cohort)):
            for j in range(i+1, len(cohort)):
                distances.append(DistanceCheck.calculate_distance(cohort[i], cohort[j], method))
        return np.array(distances).mean()

    @staticmethod
    def calculate_between_group_distance(cohort_a, cohort_b, method=0):
        distances = []
        for i in range(len(cohort_a)):
            for j in range(len(cohort_b)):
                distances.append(DistanceCheck.calculate_distance(cohort_a[i], cohort_b[j], method))
        return np.array(distances).mean()

    @staticmethod
    def calculate_distance(x, y, method):
        return list(DistanceCheck.methods.values())[method][0](x, y)

    def calculate_mean_distance(self, cohort, sample):
        distances = np.array([DistanceCheck.calculate_distance(sample, x, self.__method) for x in cohort])
        return distances.mean()

    def predict(self, samples):
        results = []
        for sample in samples:
            distances = []
            for model in self.__data['models']:
                distances.append(self.calculate_mean_distance(model["cohort"], sample))
            results.append(distances.index(min(distances)))
        return results

    def predict_real(self, cohort, samples):
        return [self.calculate_mean_distance(cohort, sample) for sample in samples]
