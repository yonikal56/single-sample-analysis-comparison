import json
from collections.abc import MutableMapping
import pandas as pd
import numpy as np
from modules import graph

number_to_mean = 4

def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'test_results-3.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

graphs = graph.Graph(1)

df = pd.DataFrame.from_dict(data)
df['random'] = 100 / df['cohorts']
df['distance.proportion'] = df['distance.between_groups'] / df['distance.in_group'] * 100
df['distance.absolute'] = df['distance.between_groups'] - df['distance.in_group']
sorted_df = df.sort_values(by=["m"])

result_columns = [col for col in df.keys() if col.startswith('results.')]
m_values = sorted_df['m'][::number_to_mean]
for col in result_columns:
    graphs.get_axes().plot(m_values, [np.mean(sorted_df[col][i*number_to_mean:(i+1)*number_to_mean]) for i in range(len(m_values))], label=col[8:])
    for i in df.index.values:
        old_value = round(df.at[i, col], 2)
        df.at[i, col] = str(old_value) + ' - ' + str(100 * old_value / df.at[i, 'random']) + '%'
graphs.legend()

graphs.show()
