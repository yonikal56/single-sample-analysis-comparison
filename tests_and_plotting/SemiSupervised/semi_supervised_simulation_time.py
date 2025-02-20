from modules import *
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# set constants
m = 1
num_of_samples = 100

file_path = 'semi_supervised_plot_doc_for_samples.json'
GLV.GLV.supervised = False
data = GLV.generate_models(m, 1, file_path, force=True, bound=0.2, probability=0.5)
model = GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A'])
initials = model.get_random_initials()
time=50
time_fractions=50
t = np.linspace(0, time, time_fractions)
a = model.solve_model(initials, time=time, time_fractions=time_fractions)

graphs = graph.Graph()
pops = []
for i in range(GLV.GLV.numOfPopulations):
    ab = []
    for ti in range(time_fractions):
        ab.append(a[i][ti])
    pops.append(ab)
for i in range(GLV.GLV.numOfPopulations):
    graphs.plot(t, pops[i], graphs.get_axes(), set_y_lim=False, color=None)

graphs.show()