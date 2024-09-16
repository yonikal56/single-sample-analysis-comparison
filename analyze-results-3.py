import json
from collections.abc import MutableMapping
import pandas as pd
import numpy as np
from modules import graph

import matplotlib
matplotlib.use("Qt5Agg")

number_to_mean3 = 4
number_to_mean4 = 4


def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'test_results-3.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

graphs = graph.Graph(1, 2)

df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["m"])

result_columns = [col for col in df.keys() if col.startswith('results.')]
m_values = sorted_df['m'][::number_to_mean3]
for col in result_columns:
    graphs.get_axes()[0].plot(m_values, [np.mean(sorted_df[col][i * number_to_mean3:(i + 1) * number_to_mean3]) for i in
                                         range(len(m_values))])
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'

data = {}
file_path = 'test_results-4.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["delta"])

result_columns = [col for col in df.keys() if col.startswith('results.')]
delta_values = sorted_df['delta'][::number_to_mean4]
method_labels = ['IDOA', 'NN', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2']
count = 0
for col in result_columns:
    graphs.get_axes()[1].plot(delta_values,
                              [np.mean(sorted_df[col][i * number_to_mean4:(i + 1) * number_to_mean4]) for i in
                               range(len(delta_values))], label=method_labels[count])
    count += 1
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'

graphs.get_plt().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
graphs.get_fig().set_figwidth(15)
graphs.get_axes()[0].set_title('a', loc='left', fontsize=15, weight='bold', x=-0.1)
graphs.get_axes()[1].set_title('b', loc='left', fontsize=15, weight='bold', x=-0.1)
graphs.get_axes()[0].set_xlabel('Cohort size, $m$', fontsize=13)
graphs.get_axes()[1].set_xlabel(r'Noise value, $\delta$', fontsize=13)
graphs.get_axes()[0].set_ylabel('Success rate', fontsize=13)
graphs.get_axes()[1].set_ylabel('Success rate', fontsize=13)
graphs.get_axes()[0].xaxis.set_tick_params(labelsize=13)
graphs.get_axes()[0].yaxis.set_tick_params(labelsize=13)
graphs.get_axes()[1].xaxis.set_tick_params(labelsize=13)
graphs.get_axes()[1].yaxis.set_tick_params(labelsize=13)

graphs.show()
