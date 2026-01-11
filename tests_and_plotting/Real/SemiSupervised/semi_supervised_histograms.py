from modules import *
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib
#matplotlib.use('TkAgg')

# set constants

data_set = f'.'
file_path = f'{data_set}/for_preprocess.csv'
metadata_path = f'{data_set}/metadata.csv'
df = pd.read_csv(file_path)
meta_data_df = pd.read_csv(metadata_path)
samples = {}
labels = {}


healthy_count = 0
sick_count = 0
for index, row in meta_data_df.iterrows():
    labels[row['ID']] = row['Tag']  # 0 for healthy, 1 for sick
    if row['Tag'] == 0:
        healthy_count += 1
    elif row['Tag'] == 1:
        sick_count += 1


for index, row in df.iterrows():
    list = row.tolist()
    if (list[0].startswith('SRR')):
        samples[list[0]] = list[1:]
        samples[list[0]] = [float(a) for a in samples[list[0]]]


hit_numbers = []
species = []
keys_to_remove = []
for key, sample in samples.items():
    hit_numbers.append(sum(sample))
    species.append(sum([(1 if a != 0 else 0) for a in sample]))


for key in keys_to_remove:
    del samples[key]
    del labels[key]

healthy_cohort = []
for key, label in labels.items():
    if key not in samples.keys():
        continue
    if label == 0:  # healthy
        healthy_cohort.append(np.array(samples[key]))


IDOA.IDOA.real = True
GLV.GLV.numOfPopulations = len(healthy_cohort[0])

m = len(healthy_cohort)
num_of_samples = int(m / 2)
shuffled_cohort = [np.array([random.choice(healthy_cohort)[i] for i in range(GLV.GLV.numOfPopulations)]) for _ in range(num_of_samples)]

data = {
    'models': [{
        'cohort': healthy_cohort[:100],
    },
    {
        'cohort': shuffled_cohort
    }]
}
data['models'][0]['cohort'] = [sample/sum(sample) for sample in data['models'][0]['cohort']]
data['models'][1]['cohort'] = [sample/sum(sample) for sample in data['models'][1]['cohort']]
samples = healthy_cohort[100:]
shuffled_samples = [np.array([random.choice(healthy_cohort)[i] for i in range(GLV.GLV.numOfPopulations)]) for _ in range(40)]
samples = samples + shuffled_samples
samples = [sample/sum(sample) for sample in samples]
real = [0] * 40 + [1] * 40


network = NeuralNetwork.NeuralNetwork(data)
idoa = IDOA.IDOA(data)
network_impact = NetworkImpact.NetworkImpact(data)
distance_check = DistanceCheck.DistanceCheck(data)
distance_check2 = DistanceCheck.DistanceCheck(data, 1)
random_forest = RandomForest.RandomForest(data)

# predictions
network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

methods = [idoa, distance_check, network_impact3]

method_labels = ['IDOA', 'DIS - BC', 'NI - WD2']

all_results = {}

roc = ROC.ROC(False)

graphs = graph.Graph(3, 1)
for i in range(len(methods)):
    axes = graphs.get_axes()[i % 3]
    predictions = methods[i].predict_real(data['models'][0]['cohort'], np.array(samples))
    graphs.hist([predictions[i] for i in range(len(predictions)) if real[i] == 0], axes, label="Real")
    graphs.hist([predictions[i] for i in range(len(predictions)) if real[i] == 1], axes, label="Shuffled")
    axes.ticklabel_format(axis="x", style="sci", scilimits=(1,2))
    axes.set_xlabel(method_labels[i], fontsize=10)
    axes.set_yticks([])
    if i == 0:
        axes.legend(loc='upper left')
        axes.set_ylabel('Probability', fontsize=10)
graphs.get_plt().subplots_adjust(left=0.1,
                                 bottom=0.1,
                                 right=0.9,
                                 top=0.9,
                                 wspace=0.2,
                                 hspace=0.5)
graphs.get_fig().set_figwidth(11)
graphs.get_fig().set_figheight(8)
print(all_results)
graphs.get_plt().savefig('../../../../article figures/5-real-hist.png')
graphs.show()
