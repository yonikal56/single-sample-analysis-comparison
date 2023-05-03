import json

import matplotlib.pyplot as plt
import numpy as np
from modules import graph

data = {}
file_path = 'test_shuffled_results.json'
with open(file_path) as file:
    data = json.load(file)

graphs = graph.Graph(3)
graphs.get_fig().set_size_inches(8, 8)
graphs.get_fig().tight_layout(pad=2)
i = 0
method_colors = {}
for key in data.keys():
    graphs.get_axes()[i].title.set_text(f'Test param - {key}')
    graphs.get_axes()[i].set_ylim(0, 1.1)
    x = data[key].keys()
    y_methods = {}
    for x_value in x:
        for method in data[key][x_value]:
            if method not in method_colors.keys():
                method_colors[method] = np.random.rand(3, )
            if method in y_methods.keys():
                y_methods[method].append(data[key][x_value][method])
            else:
                y_methods[method] = [data[key][x_value][method]]

    for method in y_methods:
        new_x = [float(x1) for x1 in x]
        label = f'{method}'
        graphs.plot(new_x, y_methods[method], graphs.get_axes()[i], label=label, color=method_colors[method], set_y_lim=False)

    i += 1

graphs.legend(5)
graphs.show()
