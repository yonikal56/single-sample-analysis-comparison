from modules import *
import numpy as np
import json
import random
import copy

all_results = []
cohorts = 2


def run_test(data, m, samples, real):
    part_data = copy.deepcopy(data)
    for j in range(cohorts):
        part_data['models'][j]['cohort'] = random.sample(part_data['models'][j]['cohort'], m)
    network = NeuralNetwork.NeuralNetwork(part_data)
    idoa = IDOA.IDOA(part_data)
    network_impact = NetworkImpact.NetworkImpact(part_data)
    distance_check = DistanceCheck.DistanceCheck(part_data)
    distance_check2 = DistanceCheck.DistanceCheck(part_data, 1)

    # predictions
    network_impact_predictions = network_impact.predict(samples)
    network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
    network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
    network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)

    methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3]

    in_group_distances = []
    between_groups_distances = []
    for i in range(cohorts):
        distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(part_data['models'][i]['cohort'])
        in_group_distances.append(distance)
    for i in range(cohorts):
        for j in range(i + 1, cohorts):
            distance = DistanceCheck.DistanceCheck.calculate_between_group_distance(part_data['models'][i]['cohort'],
                                                                                    part_data['models'][j]['cohort'])
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
        predictions = method.predict(np.array(samples))
        for prediction, re in zip(predictions, real):
            if re == prediction:
                num_of_success += 1
        success_rate = (num_of_success / len(real)) * 100
        test_results['results'][str(method)] = success_rate
    print(f'test results - {test_results}\n')
    return test_results


num_of_runs = 1
m_values = [100] * 10
num_of_samples = 100
bound = 0.025
probability = 0.1
initial_samples = 200

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(initial_samples, cohorts, file_path, bound=bound, probability=probability)

samples, real = GLV.generate_random_samples(data, num_of_samples)
iteration_num = 1
for m in m_values:
    print(iteration_num)
    iteration_num += 1
    tests_results = []
    for _ in range(num_of_runs):
        print(f'm: {m}')
        tests_results.append(run_test(data, m, samples, real))
    all_results += tests_results
print(all_results)
file_path = 'test_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)
