from modules import GLV, NetworkImpact, NeuralNetwork, DistanceCheck, RandomForest, IDOA
import numpy as np


class Testing:
    def __int__(self):
        pass

    @staticmethod
    def run_test(file_path, cohorts, m, num_of_samples, data=None, bound=0.025, probability=0.1, delta=None):
        # create samples if not given
        if data is None:
            data = GLV.generate_models(m, cohorts, file_path, bound=bound, probability=probability, force=True)
        if delta is not None:
            GLV.GLV.delta = delta

        network = NeuralNetwork.NeuralNetwork(data)
        idoa = IDOA.IDOA(data)
        network_impact = NetworkImpact.NetworkImpact(data)
        distance_check = DistanceCheck.DistanceCheck(data)
        distance_check2 = DistanceCheck.DistanceCheck(data, 1)
        random_forest = RandomForest.RandomForest(data)

        # predictions
        samples, real = GLV.generate_random_samples(data, num_of_samples)
        network_impact_predictions = network_impact.predict(samples)

        network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
        network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
        network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
        network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
        network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

        methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2,
                   network_impact3,
                   network_impact4, network_impact5, random_forest]

        states = []

        in_group_distances = []
        between_groups_distances = []
        for i in range(cohorts):
            states.extend(data['models'][i]['cohort'])
            distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(data['models'][i]['cohort'])
            in_group_distances.append(distance)
        for i in range(cohorts):
            for j in range(i + 1, cohorts):
                distance = DistanceCheck.DistanceCheck.calculate_between_group_distance(data['models'][i]['cohort'],
                                                                                        data['models'][j]['cohort'])
                between_groups_distances.append(distance)

        test_results = {
            'm': m,
            'cohorts': cohorts,
            'tests': len(real),
            'nn_accuracy': network.get_accuracy(),
            'distance': {
                'in_group': np.array(in_group_distances).mean(),
                'between_groups': np.array(between_groups_distances).mean()
            },
            'results': {

            }
        }

        if delta is not None:
            test_results['delta'] = delta

        for method in methods:
            num_of_success = 0
            predictions = method.predict(np.array(samples))
            results = []
            for prediction, re in zip(predictions, real):
                if re == prediction:
                    num_of_success += 1
                results.append((prediction, re, re == prediction))
            success_rate = (num_of_success / len(real)) * 100
            test_results['results'][str(method)] = success_rate
        print(test_results)
        return test_results

    @staticmethod
    def get_methods_names():
        return ['IDOA', 'NN', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2', 'RF']