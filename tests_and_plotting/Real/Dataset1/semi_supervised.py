import sys
import multiprocessing
import pandas as pd
from modules import *
import numpy as np
import json
from pathos.multiprocessing import ProcessingPool as Pool

# Load the CSV file
file_path = 'OTU1.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

group_1_prefix = 'A'
group_2_prefix = 'B'

# Filter columns based on the first letter
group_1_columns = [col for col in df.columns if col.startswith(group_1_prefix)]
group_2_columns = [col for col in df.columns if col.startswith(group_2_prefix)]

# Create lists of vectors for each group
group_1_vectors = np.array([np.array(df[col].tolist())/sum(df[col].tolist()) for col in group_1_columns])
group_2_vectors = np.array([np.array(df[col].tolist())/sum(df[col].tolist()) for col in group_2_columns])


# Define the function to run a single test
def run_single_test(args):
    return 5
    #return Testing.Testing.run_test_semi_supervised_real_data(group_2_vectors, group_1_vectors, number_of_runs=1)


# Number of runs
number_of_runs = 10

results = Testing.Testing.run_test_semi_supervised_real_data(group_2_vectors, group_1_vectors, number_of_runs=number_of_runs)
# Save results to a JSON file
file_path = 'semi_supervised_auc_results.json'
with open(file_path, 'w') as outfile:
    json.dump(results, outfile)

print(results)
