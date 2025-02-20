from modules import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# set constants
m = 100
num_of_samples = 100

file_path = 'semi_supervised_histograms_samples.json'
GLV.GLV.supervised = False
data = GLV.generate_models(m, 1, file_path, force=True, bound=0.15)
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
random_forest = RandomForest.RandomForest(data)

# predictions
network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
           network_impact4, network_impact5, random_forest]

method_labels = Testing.Testing.get_methods_names()

all_results = {}

roc = ROC.ROC(False)

graphs = graph.Graph(4, 3)
for i in range(len(methods)):
    axes = graphs.get_axes()[i // 3][i % 3]
    predictions = methods[i].predict_real(data['models'][0]['cohort'], np.array(samples))
    auc = roc.add_graph(real, predictions, str(methods[i]))
    all_results[method_labels[i]] = auc
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
graphs.show()
