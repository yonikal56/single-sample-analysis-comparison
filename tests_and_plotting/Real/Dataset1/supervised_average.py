import json
from collections.abc import MutableMapping
import pandas as pd
import numpy as np
from modules import graph, Testing

def flatten_dict(d: MutableMapping, sep: str = '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict


data = {}
file_path = 'supervised_shuffled_results.json'
with open(file_path) as file:
    data = [flatten_dict(test) for test in json.load(file)]

df = pd.DataFrame.from_dict(data)
result_columns = [col for col in df.keys() if col.startswith('results.')]
results = {}
maxes = {}
for result_column in result_columns:
    results[result_column] = df[result_column].mean()
    maxes[result_column] = df[result_column].max()
print(f'mean: {results}\n maxes: {maxes}')
