from modules import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

m = 100
num_of_samples = 200

file_path = 'samples_.json'
data = GLV.generate_models(m, 1, file_path, force=True, bound=0., probability=0)
samples, real = GLV.generate_random_shuffled_samples(data['models'][0], num_of_samples)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(samples)

true = [principalComponents[i] for i in range(num_of_samples) if real[i] == 0]
shuffled = [principalComponents[i] for i in range(num_of_samples) if real[i] == 1]
network_impact = NetworkImpact.NetworkImpact(data)
network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)

# graphs = graph.Graph(1)
# graphs.scatter([x[0] for x in true], [x[1] for x in true], graphs.get_axes(), color='green', fit=False, set_y_lim=False)
# graphs.scatter([x[0] for x in shuffled], [x[1] for x in shuffled], graphs.get_axes(), color='red', fit=False, set_y_lim=False)

# graphs.show()
graphs = graph.Graph(4)

plt.subplots_adjust(hspace=1)

params = [network_impact1, network_impact2, network_impact3, network_impact4]
for n in range(4):
    res = params[n].predict_real(data['models'][0]['cohort'], samples)
    true = [res[i] for i in range(num_of_samples) if real[i] == 0]
    shuffled = [res[i] for i in range(num_of_samples) if real[i] == 1]
    graphs.get_axes()[n].title.set_text(str(params[n]))
    graphs.scatter(true, np.linspace(0, 1, len(true)), graphs.get_axes()[n], color='green', fit=False)
    graphs.scatter(shuffled, np.linspace(0, 1, len(shuffled)), graphs.get_axes()[n], color='red', fit=False)
graphs.show()
