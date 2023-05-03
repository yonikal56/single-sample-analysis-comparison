from modules import *
import numpy as np
import json

num_of_samples = 100

base_params = {
    'm': 100,
    'bound': 0.025,
    'probability': 0.1
}


def run_test(param_name, param):
    params = base_params.copy()
    params[param_name] = param

    params['m'] = int(params['m'])

    file_path = 'samples2.json'
    data = GLV.generate_models(params['m'], 1, file_path, bound=params['bound'], probability=params['probability'])
    data['models'].append({
        'r': data['models'][0]['r'],
        'A': data['models'][0]['A'],
        'cohort': GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A']).get_shuffled_samples(params['m'],
                                                                                                   data['models'][0][
                                                                                                       'cohort'])
    })
    samples, real = GLV.generate_random_shuffled_samples(data['models'][0], num_of_samples)

    network = NeuralNetwork.NeuralNetwork(data)
    idoa = IDOA.IDOA(data)
    network_impact = NetworkImpact.NetworkImpact(data)
    distance_check = DistanceCheck.DistanceCheck(data)
    distance_check2 = DistanceCheck.DistanceCheck(data, 1)

    # predictions
    network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
    network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
    network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
    network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
    network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)

    methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
               network_impact4]

    roc = ROC.ROC(plot=False)  # don't plot, only calculate auc

    results = {}

    for method in methods:
        predictions = method.predict_real(data['models'][0]['cohort'], np.array(samples))
        auc = roc.add_graph(real, predictions, str(method))
        results[str(method)] = auc

    return results


m_values = np.linspace(20, 200, num=5)
bound_values = np.linspace(0.0, 0.025, num=5)
probability_values = np.linspace(0, 1, num=5)
all_tests = {
    'm': m_values,
    'bound': bound_values,
    'probability': probability_values
}

all_results = {
    'm': {},
    'bound': {},
    'probability': {}
}

for test in all_tests.keys():
    print(f'test {test}\n-------------------')
    i = 1
    for p in all_tests[test]:
        print(f'test {test} - num {i}')
        result = run_test(test, p)
        all_results[test][p] = result
        i += 1
    print('-------------------\n')

print(all_results)
file_path = 'test_shuffled_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)
