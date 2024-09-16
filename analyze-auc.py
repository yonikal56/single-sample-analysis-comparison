import json
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Qt5Agg")

data = {}
file_path = 'test_results-auc.json'
with open(file_path) as file:
    data = json.load(file)

box_plot_data = []
labels = []
for method in reversed(data.keys()):
    labels.append(method)
    box_plot_data.append(data[method])

fig = plt.figure(figsize =(11, 8))
ax = fig.add_subplot(111)
ax.boxplot(box_plot_data,patch_artist=True,labels=labels, vert = 0)
ax.set_xlabel('AUC', fontsize=13)
plt.show()
