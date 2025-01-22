import json
import matplotlib.pyplot as plt
import pandas as pd
from modules import *
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

# Load the CSV file
file_path = 'OTU1.xlsx'  # Replace with your CSV file path
df = pd.read_excel(file_path)

group_1_prefix = 'H'
group_2_prefix = 'G'

# Filter columns based on the first letter
group_1_columns = [col[:-2] for col in df.columns if col.startswith(group_1_prefix) and col.endswith('.1')]
group_2_columns = [col[:-2] for col in df.columns if col.startswith(group_2_prefix) and col.endswith('.1')]

samples = group_1_columns + group_2_columns

data = {}
file_path = 'semi_supervised_auc_results.json'
with open(file_path) as file:
    data = json.load(file)

print(len(data[0]['IDOA']))

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.bar(samples, data[0]['IDOA'])
ax2.bar(samples, data[0]['NI - SD'])
plt.show()
