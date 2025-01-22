import pandas as pd
from modules import *
import numpy as np
import json
import random
from modules.progressbar import ProgressBar

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


def get_shuffled_sample(cohort):
    # copy cohort
    samples = cohort.copy()
    # each index is shuffled sample is a value from random sample inside cohort in the same index
    sample = np.array([random.choice(samples)[i] for i in range(len(cohort[0]))])
    return sample


def get_shuffled_samples(m, cohort):
    # create m shuffled samples
    samples = []
    progress = ProgressBar(m, f'creating {str(m)} shuffled samples', 25)
    for i in range(m):
        samples.append(get_shuffled_sample(cohort))
        progress.update()
    del progress
    return samples


shuffled_cohort_health = get_shuffled_samples(len(group_2_vectors), group_2_vectors)
shuffled_cohort_asd = get_shuffled_samples(len(group_1_vectors), group_1_vectors)

shuffled_cohort_health = [a/sum(a) for a in shuffled_cohort_health]
shuffled_cohort_asd = [a/sum(a) for a in shuffled_cohort_asd]

data = {
    'models':
        [
            {'cohort': np.array(shuffled_cohort_health)},
            {'cohort': np.array(shuffled_cohort_asd)}
        ]
}

num_of_runs = 1
all_results = []
for i in range(num_of_runs):
    print(f'run number {i+1}')
    results = Testing.Testing.run_test('', 2, 0, 0, data)
    all_results.append(results)

print(all_results)
file_path = 'supervised_shuffled_results.json'
with open(file_path, 'w') as outfile:
    json.dump(all_results, outfile)
