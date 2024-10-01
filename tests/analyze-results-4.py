import json
from collections.abc import MutableMapping
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Qt5Agg")

number_to_mean = 4

def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = '../test_results-5.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]
pca_data = {}

file_path = '../test_results-5-pca.json'
with open(file_path) as file:
    pca_data = json.load(file)

str = "ABCD;EEEE"
fig, axes = plt.subplot_mosaic(str, constrained_layout=True)

full_str = "aa" + str
m = pca_data["m"]

nn = 0
for pca_graph in pca_data["data"]:
    # initialise the standard scaler
    sc = StandardScaler()
    # set the components to 2
    pca = PCA(n_components=2, whiten=True)

    cohorts_num = pca_graph["cohorts"]
    states = pca_graph["points"]

    df = pd.DataFrame.from_dict(states)

    df_scaled = pd.DataFrame(sc.fit_transform(df), columns=df.columns)

    pca.fit(df_scaled)

    # fit the model to our data and extract the results
    X_pca = pca.transform(df_scaled)

    # create a dataframe from the dataset
    df = pd.DataFrame(data=X_pca,
                      columns=["Component1",
                               "Component2"])

    ax = axes[full_str[cohorts_num]]
    for i in range(cohorts_num):
        ax.scatter(df['Component1'].iloc[i*m:(i+1)*m], df['Component2'].iloc[i*m:(i+1)*m], s=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    if cohorts_num == 2:
        ax.set_xlabel('PC1', fontsize=13)
        ax.set_ylabel('PC2', fontsize=13)
        ax.xaxis.set_label_position('top')

    nn += 1

for c in "ABCD":
    axes[c].set_title(c.lower(), loc='left', fontsize=15, weight='bold', x=-0.1, y=1)
axes['E'].set_title('e', loc='left', fontsize=15, weight='bold', x=-0.025, y=1)


df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["cohorts"])

result_columns = [col for col in df.keys() if col.startswith('results.')]
cohorts_values = sorted_df['cohorts'][::number_to_mean]
ax = axes["E"]
ax.set_xlabel('Number of cohorts', fontsize=13)
ax.set_ylabel('Success rate', fontsize=13)
ax.xaxis.set_tick_params(labelsize=13)
ax.yaxis.set_tick_params(labelsize=13)
ax.set_ylim(0, 110)
ax.set_xticks(cohorts_values)
method_labels = ['IDOA', 'NN', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2']

count = 0
for col in result_columns:
    ax.scatter(cohorts_values, [np.mean(sorted_df[col][i*number_to_mean:(i+1)*number_to_mean]) for i in range(len(cohorts_values))], label=method_labels[count])
    ax.plot(cohorts_values, [np.mean(sorted_df[col][i*number_to_mean:(i+1)*number_to_mean]) for i in range(len(cohorts_values))], linestyle='dashed', linewidth=0.5)
    count += 1

ax.scatter(cohorts_values, [100 / cohort_value for cohort_value in cohorts_values], label="Random", c="black", marker="_", s=200)

fig.set_figwidth(15)
fig.set_figheight(7)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), prop={'size': 10}, fancybox=True, ncol=10, edgecolor="gray")
plt.show()
