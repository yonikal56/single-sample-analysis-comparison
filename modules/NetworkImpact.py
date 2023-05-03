from modules.GLV import GLV
from modules.DOC import DOC
from modules.progressbar import ProgressBar
import numpy as np
import math
from scipy import stats


class NetworkImpact:
    def __init__(self, data):
        self.__data = data
        self.__networks = []
        self.__bounds = None
        self.__methods = [self.predict_structural_difference, self.predict_weight_difference,
                          self.predict_origin_weight_difference,
                          self.predict_theta,
                          self.predict_origin_theta]
        for model in self.__data['models']:
            self.__networks.append(self.calculate_network(np.array(model['cohort'])))
            print(self.__networks[0])

    def calculate_bounds(self, cohort):
        m = len(cohort)
        progress = ProgressBar(m, f'calculating {str(m)} samples for upper boundary', 50)
        params = []
        cohort = np.array(cohort)
        network_with = self.calculate_network(cohort)
        for i in range(m):
            network_without = self.calculate_network(np.append(cohort[:i], cohort[i + 1:], axis=0))
            params.append([method(network_with, network_without) for method in self.__methods])
            progress.update()
        del progress
        self.__bounds = []
        for k in range(len(self.__methods)):
            param_values = [param[k] for param in params]
            param_values.sort()
            bound = round(len(cohort) * 0.95)
            self.__bounds.append(param_values[bound - 1])

    @staticmethod
    def unweighted_jaccard_similarity(x, y):
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = intersection.sum() / float(union.sum())
        return similarity

    @staticmethod
    def calculate_pearson_correlation_p_value(samples, i, j):
        shared_samples = samples[(samples[:, i] > DOC.epsilon) & (samples[:, j] > DOC.epsilon)]
        result = stats.pearsonr(shared_samples[:, i], shared_samples[:, j])
        return result[0], result[1]

    def calculate_network(self, samples):
        n = GLV.numOfPopulations
        network = [[1 if j == i else 0 for j in range(n)] for i in range(n)]
        unweighted_network = [[1 if j == i else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                pearson_correlation, p_value = self.calculate_pearson_correlation_p_value(samples, i, j)
                network[i][j] = pearson_correlation if p_value < math.pow(10, -3) else 0
                unweighted_network[i][j] = 1 if p_value < math.pow(10, -3) else 0
        return network, unweighted_network

    def predict_structural_difference(self, network_a, network_b):
        return 1 - self.unweighted_jaccard_similarity(np.array(network_a[1]), np.array(network_b[1]))

    def predict_weight_difference(self, network_a, network_b):
        s = 0
        count = 0
        n = GLV.numOfPopulations
        for i in range(n):
            for j in range(i + 1, n):
                if abs(network_a[0][i][j]) > DOC.epsilon and abs(network_b[0][i][j]) > DOC.epsilon:
                    # add the absolute difference
                    s += abs(network_a[0][i][j] - network_b[0][i][j])
                    count += 1
        return s / count

    def predict_origin_weight_difference(self, network_a, network_b):
        s = 0
        count = 0
        n = GLV.numOfPopulations
        for i in range(n):
            for j in range(i + 1, n):
                if abs(network_a[0][i][j]) > DOC.epsilon and abs(network_b[0][i][j]) > DOC.epsilon:
                    s += network_a[0][i][j] - network_b[0][i][j]
                    count += 1
        return s / count

    def predict_theta(self, network_a, network_b):
        increased_weight_count = 0
        decreased_weight_count = 0
        n = GLV.numOfPopulations
        for i in range(n):
            for j in range(i + 1, n):
                if DOC.epsilon < abs(network_a[0][i][j]) and abs(network_b[0][i][j]) > DOC.epsilon \
                        and network_a[0][i][j] != network_b[0][i][j]:
                    if network_b[0][i][j] < network_a[0][i][j]:
                        decreased_weight_count += 1
                    else:
                        increased_weight_count += 1
        # compare decreased / increased to 1
        val = (decreased_weight_count + DOC.epsilon) / (increased_weight_count + DOC.epsilon)
        return abs(1 - val)

    def predict_origin_theta(self, network_a, network_b):
        different_weight_count = 0
        decreased_weight_count = 0
        n = GLV.numOfPopulations
        for i in range(n):
            for j in range(i + 1, n):
                if DOC.epsilon < abs(network_a[0][i][j]) and abs(network_b[0][i][j]) > DOC.epsilon \
                        and network_a[0][i][j] != network_b[0][i][j]:
                    different_weight_count += 1
                    if network_b[0][i][j] < network_a[0][i][j]:
                        decreased_weight_count += 1
        val = (decreased_weight_count + DOC.epsilon) / (different_weight_count + DOC.epsilon)
        return abs(0.5 - val)

    def predict(self, samples):
        method_indexes = [0, 1, 2, 3, 4]
        methods = [self.__methods[i] for i in method_indexes]
        return self.calculate_prediction(samples, methods)

    def predict_real(self, cohort, samples):
        method_indexes = [0, 1, 2, 3, 4]
        methods = [self.__methods[i] for i in method_indexes]
        results = []
        m = len(samples)
        progress = ProgressBar(m, f'evaluating {str(m)} samples with network impact method', 50)
        network_without = self.calculate_network(np.array(cohort))
        for n in range(m):
            sample = samples[n]
            network_with = self.calculate_network(np.append(cohort, [sample], axis=0))
            params = [method(network_without, network_with) for method in methods]
            predictions = [params[k] for k in range(len(methods))]
            results.append(predictions)
            progress.update()
        del progress
        return results

    def calculate_prediction(self, samples, methods):
        results = []
        m = len(samples)
        progress = ProgressBar(m, f'evaluating {str(m)} samples with network impact method', 50)
        for n in range(m):
            sample = samples[n]
            params = []
            for i in range(len(self.__data['models'])):
                network_with = self.calculate_network(np.append(self.__data['models'][i]['cohort'], [sample], axis=0))
                params.append([method(self.__networks[i], network_with) for method in methods])
            predictions = []
            for k in range(len(methods)):
                param_values = [param[k] for param in params]
                predictions.append(param_values.index(min(param_values)))
            results.append(predictions)
            progress.update()
        del progress
        return results


class NetworkImpactHandler:
    def __init__(self, predictions, method):
        self.__predictions = np.array(predictions)
        self.__method = method

    def predict(self, samples):
        return self.__predictions[:, self.__method]

    def predict_real(self, cohort, samples):
        return self.predict(samples)

    def __str__(self):
        types = ['structural difference', 'weight difference', 'origin weight difference', 'theta', 'origin theta']
        return f'Network Impact - {types[self.__method]}'
