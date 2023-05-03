from modules import *
import numpy as np
import json
import itertools

all_results = []

def run_test(cohorts, m, num_of_samples, bound=0.025, probability=0.1):
    # create two different GLV models with m samples
    file_path = 'samples.json'
    data = GLV.generate_models(m, cohorts, file_path, bound=bound, probability=probability, force=True)

    network = NeuralNetwork.NeuralNetwork(data)
    idoa = IDOA.IDOA(data)
    network_impact = NetworkImpact.NetworkImpact(data)
    distance_check = DistanceCheck.DistanceCheck(data)
    distance_check2 = DistanceCheck.DistanceCheck(data, 1)

    # predictions
    samples, real = GLV.generate_random_samples(data, num_of_samples)
    network_impact_predictions = network_impact.predict(samples)

    network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
    network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
    network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)

    methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3]

    in_group_distances = []
    between_groups_distances = []
    for i in range(cohorts):
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
        'distance': {
            'in_group': np.array(in_group_distances).mean(),
            'between_groups': np.array(between_groups_distances).mean()
        },
        'results': {

        }
    }

    for method in methods:
        num_of_success = 0
        print(f'method:{str(method)}')
        predictions = method.predict(np.array(samples))
        results = []
        for prediction, re in zip(predictions, real):
            if re == prediction:
                num_of_success += 1
            results.append((prediction, re, re == prediction))
        print(results)
        success_rate = (num_of_success / len(real)) * 100
        print(f'success_rate : {success_rate}')
        print('------------------------')
        test_results['results'][str(method)] = success_rate
    print(f'test results - {test_results}\n\n')
    return test_results


cohort_values = [2]
num_of_runs = 1
m_values = range(30, 41, 10)
samples_values = [100]
bound = 0.025
probability = 0.1

tests = list(itertools.product(cohort_values, m_values, samples_values))
for cohorts, m, num_of_samples in tests:
    tests_results = []
    for _ in range(num_of_runs):
        print(f'cohorts: {cohorts}, m: {m}, samples: {num_of_samples}')
        tests_results.append(run_test(cohorts, m, num_of_samples, bound, probability))
    all_results += tests_results
print(all_results)
file_path = 'test_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)
