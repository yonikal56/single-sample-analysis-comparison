from modules import *
import numpy as np

all_results = []


num_of_runs = 1
m_values = [50] * 30
num_of_samples = 100
bound = 0.025
probability = 0.1
initial_samples = 100

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(initial_samples, 1, file_path, bound=bound, probability=probability)
data['models'].append({
    'r': data['models'][0]['r'],
    'A': data['models'][0]['A'],
    'cohort': GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A']).get_shuffled_samples(initial_samples, data['models'][0]['cohort'])
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

methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3]
test_results = {}

for method in methods:
    num_of_success = 0
    predictions = method.predict_real(data['models'][0]['cohort'], np.array(samples))
    for prediction, re in zip(predictions, real):
        if re == prediction:
            num_of_success += 1
    success_rate = (num_of_success / len(real)) * 100
    test_results[str(method)] = success_rate
print(f'test results - {test_results}\n')
