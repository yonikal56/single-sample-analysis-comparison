import pandas as pd
from modules import *
import numpy as np
import json

# Load the CSV file
file_path = 'OTU1.xlsx'  # Replace with your CSV file path
df = pd.read_excel(file_path)

group_1_prefix = 'H'
group_2_prefix = 'G'

# Filter columns based on the first letter
group_1_columns = [col for col in df.columns if col.startswith(group_1_prefix) and col.endswith('.1')]
group_2_columns = [col for col in df.columns if col.startswith(group_2_prefix) and col.endswith('.1')]

# Create lists of vectors for each group
group_1_vectors = np.array([(np.array(df[col].tolist())/sum(df[col].tolist()))[:-1] for col in group_1_columns])
group_2_vectors = np.array([(np.array(df[col].tolist())/sum(df[col].tolist()))[:-1] for col in group_2_columns])


results = Testing.Testing.run_test_semi_supervised_drop_one_out(group_1_vectors, group_2_vectors)
print(results)
file_path = 'semi_supervised_auc_results.json'
with open(file_path, 'w') as outfile:
    json.dump(results, outfile)

