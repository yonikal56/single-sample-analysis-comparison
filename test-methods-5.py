from modules import *
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
import pandas as pd

all_results = []

cohort_values = [2, 3, 4]
num_of_runs = 1
m_values = [100]
samples_values = [100]
bound = 0.025
probability = 0.1

tests = list(itertools.product(cohort_values, m_values, samples_values))

graphs = graph.Graph(2, len(tests))

x = ['IDOA', 'NN', 'Bray-Curtis', 'Euclidean', 'NI - SD', 'NI - WD', 'NI - OWD', 'NI - T', 'NI - OT']
colors = np.random.rand(9, 3)

def run_test(cohorts, m, num_of_samples, iteration_number, bound=0.025, probability=0.1):
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
    network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
    network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

    methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
               network_impact4, network_impact5]

    # initialise the standard scaler
    sc = StandardScaler()
    # set the components to 2
    pca = PCA(n_components=2, whiten=True)
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

    # plot PCA
    df = pd.DataFrame.from_dict(states)

    df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)

    pca.fit(df_scaled)

    # fit the model to our data and extract the results
    X_pca = pca.transform(df_scaled)

    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_pca,
                      columns=["Component1",
                               "Component2"])

    for i in range(cohorts):
        graphs.scatter(df['Component1'].iloc[m*i:m*(i+1) - 1], df['Component2'].iloc[m*i:m*(i+1) - 1], graphs.get_axes()[0][iteration_number], False,
                       False)

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

    y = []

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
        y.append(success_rate)
        print(f'success_rate : {success_rate}')
        print('------------------------')
        test_results['results'][str(method)] = success_rate
    print(f'test results - {test_results}\n\n')

    graphs.bar(x,y, graphs.get_axes()[1][iteration_number], colors=colors)

    return test_results

i = 0
for cohorts, m, num_of_samples in tests:
    tests_results = []
    for _ in range(num_of_runs):
        print(f'cohorts: {cohorts}, m: {m}, samples: {num_of_samples}')
        tests_results.append(run_test(cohorts, m, num_of_samples, i, bound, probability))
        i += 1
    all_results += tests_results
print(all_results)
file_path = 'test_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)

graphs.legend()
graphs.show()
