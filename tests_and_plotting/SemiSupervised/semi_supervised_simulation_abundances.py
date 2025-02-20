from modules import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# set constants
m = 100
num_of_samples = 100

file_path = 'semi_supervised_plot_doc_for_samples.json'
GLV.GLV.supervised = False
GLV.GLV.delta = 0
data = GLV.generate_models(m, 1, file_path, force=True, bound=0.2, probability=0.5)

model = GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A'])
shuffles_samples = model.get_shuffled_samples(100, data['models'][0]['cohort'])
shuffles_samples = [[a / sum(sample) for a in sample] for sample in shuffles_samples]
cohort = data['models'][0]['cohort']
cohort = [[a / sum(sample) for a in sample] for sample in data['models'][0]['cohort']]
all = cohort + shuffles_samples

graphs = graph.Graph(5)
for i in range(5):
    graphs.scatter(set_y_lim=False, x=list(range(len(all))), y=[a[i] for a in all], axs=graphs.get_axes()[i])


graphs.get_fig().set_figwidth(11)
graphs.get_fig().set_figheight(8)
graphs.show()