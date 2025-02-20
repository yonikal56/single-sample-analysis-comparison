import json
from collections.abc import MutableMapping
import pandas as pd
from modules import graph
import numpy as np


def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'gradual_classification_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

df = pd.DataFrame.from_dict(data[0]).T
sorted_df = df.sort_values(by=["delta"])
delta_values = sorted_df['delta'][:-1]
results = {}
result_columns = [col for col in df.keys() if col.startswith('results.')]
for col in result_columns:
    results[col] = np.zeros(len(delta_values))
graphs = graph.Graph(1)

for i in range(4):
    df = pd.DataFrame.from_dict(data[i]).T
    sorted_df = df.sort_values(by=["delta"])
    result_columns = [col for col in df.keys() if col.startswith('results.')]
    for col in result_columns:
        results[col] += np.array(sorted_df[col][:-1])

df = pd.DataFrame.from_dict(data[0]).T
sorted_df = df.sort_values(by=["delta"])
delta_values = sorted_df['delta'][:-1]
result_columns = [col for col in df.keys() if col.startswith('results.')]
for col in result_columns:
    graphs.get_axes().plot(sorted_df['delta'][:-1], results[col]/4, label=col[8:])
graphs.legend()

graphs.show()
