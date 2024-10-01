from modules import *
import numpy as np
import json

# set constants
m = 100
num_of_samples = 100
number_of_runs = 100

method_labels = ['IDOA', 'NN', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2']
all_results = {}
for method_label in method_labels:
    all_results[method_label] = []


def run_test():
    file_path = '../shuffled-samples-auc.json'
    data = GLV.generate_models(m, 1, file_path, force=True, bound=0.025)
    data['models'].append({
        'r': data['models'][0]['r'],
        'A': data['models'][0]['A'],
        'cohort': GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A']).get_shuffled_samples(m,
                                                                                                   data['models'][0][
                                                                                                       'cohort'])
    })
    samples, real = GLV.generate_random_shuffled_samples(data['models'][0], num_of_samples)
    data['models'][0]['cohort'] = [[a / sum(sample) for a in sample] for sample in data['models'][0]['cohort']]
    data['models'][1]['cohort'] = [[a / sum(sample) for a in sample] for sample in data['models'][1]['cohort']]
    samples = [[a / sum(sample) for a in sample] for sample in samples]

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
    network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

    methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
               network_impact4, network_impact5]

    roc = ROC.ROC(False)

    auc_r = []
    for method in methods:
        predictions = method.predict_real(data['models'][0]['cohort'], np.array(samples))
        auc = roc.add_graph(real, predictions, str(method))
        all_results[method_labels[methods.index(method)]].append(auc)
        auc_r.append(auc)
    print(auc_r)


for i in range(number_of_runs):
    print(f'--run number {i + 1} from {number_of_runs}:')
    run_test()

file_path = '../test_results-auc.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)
