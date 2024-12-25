import pandas as pd
from modules import *
import numpy as np

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


data = {
    'models':
        [
            {'cohort': group_1_vectors},
            {'cohort': group_2_vectors}
        ]
}

results = Testing.Testing.run_test('', 2, 0, 0, data)

