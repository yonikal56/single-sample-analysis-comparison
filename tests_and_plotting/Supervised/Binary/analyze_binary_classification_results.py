import json
from collections.abc import MutableMapping
import pandas as pd
import numpy as np
from modules import graph, Testing

import matplotlib.pyplot as plt

#matplotlib.use("Qt5Agg")

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)
gs.update(wspace=0.5, hspace=0.5)

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

number_to_mean3 = 4
number_to_mean4 = 4

cols_to_keep = ['results.IDOA', 'results.Neural Network', 'results.Bray-Curtis Dissimilarity', 'results.Network Impact - weight difference', 'results.Random Forest']
method_labels = ['IDOA', 'NN', 'DIS - BC', 'NI - WD1', 'RF']

def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'binary_classification_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]


df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["m"])

result_columns = cols_to_keep
m_values = sorted_df['m'][::number_to_mean3]
for col in result_columns:
    ax1.plot(m_values, [np.mean(sorted_df[col][i * number_to_mean3:(i + 1) * number_to_mean3]) for i in
                                         range(len(m_values))])
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'

data = {}
file_path = 'Noise/noise_tests_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["delta"])

result_columns = cols_to_keep
delta_values = sorted_df['delta'][::number_to_mean4]
count = 0
for col in result_columns:
    ax2.plot(delta_values,
                              [np.mean(sorted_df[col][i * number_to_mean4:(i + 1) * number_to_mean4]) for i in
                               range(len(delta_values))], label=method_labels[count])
    count += 1
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'

data = {}
file_path = 'GradualChange/gradual_classification_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

df = pd.DataFrame.from_dict(data[0]).T
df['random'] = 50
sorted_df = df.sort_values(by=["delta"])

result_columns = cols_to_keep
delta_values = sorted_df['delta'][::number_to_mean4]
count = 0
for col in result_columns:
    ax3.plot(delta_values,
                              [np.mean(sorted_df[col][i * number_to_mean4:(i + 1) * number_to_mean4]) for i in
                               range(len(delta_values))], label=method_labels[count])
    count += 1
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
ax1.set_title('a', loc='left', fontsize=15, weight='bold', x=-0.1)
ax2.set_title('b', loc='left', fontsize=15, weight='bold', x=-0.1)
ax3.set_title('c', loc='left', fontsize=15, weight='bold', x=-0.1)
ax1.set_xlabel('Cohort size, $m$', fontsize=13)
ax2.set_xlabel(r'Noise value, $\delta$', fontsize=13)
ax3.set_xlabel(r'Difference parameter, $\tau$', fontsize=13)
ax1.set_ylabel('Success rate', fontsize=13)
ax2.set_ylabel('Success rate', fontsize=13)
ax3.set_ylabel('Success rate', fontsize=13)
ax1.xaxis.set_tick_params(labelsize=13)
ax1.yaxis.set_tick_params(labelsize=13)
ax2.xaxis.set_tick_params(labelsize=13)
ax2.yaxis.set_tick_params(labelsize=13)
ax3.xaxis.set_tick_params(labelsize=13)
ax3.yaxis.set_tick_params(labelsize=13)

plt.savefig('fig2.png')
plt.show()