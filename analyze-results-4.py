import json
from collections.abc import MutableMapping
import pandas as pd
import numpy as np
from modules import graph


def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'test_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

graphs = graph.Graph(1)

df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["delta"])

print(sorted_df['delta'][::2])

result_columns = [col for col in df.keys() if col.startswith('results.')]
delta_values = sorted_df['delta'][::2]
for col in result_columns:
    graphs.get_axes().plot(delta_values, [(sorted_df[col][i*2] + sorted_df[col][i*2+1])/2 for i in range(len(sorted_df['delta'][::2]))], label=col[8:])
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'
graphs.legend()

graphs.show()
