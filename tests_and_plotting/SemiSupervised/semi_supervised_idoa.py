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
data = GLV.generate_models(m, 1, file_path, force=False, bound=0.2, probability=0.5)

model = GLV.GLV(r=data['models'][0]['r'], A=data['models'][0]['A'])
null_glv_sample = GLV.GLV(r=data['models'][0]['r'], A=np.diag([-1] * GLV.GLV.numOfPopulations)).get_samples(1)
#null_glv_sample = [[a / sum(sample) for a in sample] for sample in null_glv_sample]
shuffles_samples = model.get_shuffled_samples(1, data['models'][0]['cohort'])
#shuffles_samples = [[a / sum(sample) for a in sample] for sample in shuffles_samples]
cohort = data['models'][0]['cohort']
cohort = [[a / sum(sample) for a in sample] for sample in data['models'][0]['cohort']]

graphs = graph.Graph(1, 2)

doc = DOC.DOC()
doc_points = [doc.get_dissimilarity_overlap_point(null_glv_sample[0], co) for co in cohort]
x = []
y = []
for dis, ov in doc_points:
    if 0.99 > ov >= 0.5:
        x.append(ov)
        y.append(dis)
x = np.array(x)
y = np.array(y)
a, b = np.polyfit(x, y, 1)
graphs.get_axes()[0].plot(x, a * x + b, color='red', linestyle='--', linewidth=2, label=f'null: {a}')
graphs.get_axes()[0].scatter(x, y)
graphs.get_axes()[0].legend()

doc = DOC.DOC()
doc_points = [doc.get_dissimilarity_overlap_point(shuffles_samples[0], co) for co in cohort]
x = []
y = []
for dis, ov in doc_points:
    if 0.99 > ov >= 0.5:
        x.append(ov)
        y.append(dis)
x = np.array(x)
y = np.array(y)
a, b = np.polyfit(x, y, 1)
graphs.get_axes()[1].plot(x, a * x + b, color='red', linestyle='--', linewidth=2, label=f'shuffled: {a}')
graphs.get_axes()[1].scatter(x, y)
graphs.get_axes()[1].legend()


graphs.get_fig().set_figwidth(11)
graphs.get_fig().set_figheight(8)
graphs.show()