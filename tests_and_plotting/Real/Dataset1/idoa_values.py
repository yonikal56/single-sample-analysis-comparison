import pandas as pd
from modules import *
import numpy as np
import random
import matplotlib.pyplot as plt

file_path = 'OTU1.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

group_1_prefix = 'A'
group_2_prefix = 'B'

# Filter columns based on the first letter
group_1_columns = [col for col in df.columns if col.startswith(group_1_prefix)]
group_2_columns = [col for col in df.columns if col.startswith(group_2_prefix)]

# Create lists of vectors for each group
healthy_cohort = np.array([(np.array(df[col].tolist())/sum(df[col].tolist()))[:-1] for col in group_2_columns])
other_cohort = np.array([(np.array(df[col].tolist())/sum(df[col].tolist()))[:-1] for col in group_1_columns])

IDOA.IDOA.real = False
GLV.GLV.numOfPopulations = len(healthy_cohort[0])

IDOA_values = []
for i in range(len(other_cohort)):
    cohorts_list = [healthy_cohort]

    models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
    data = {
        'models': models
    }
    idoa = IDOA.IDOA(data)
    IDOA_values.append(idoa.calculate_IDOA(healthy_cohort, other_cohort[i]))

plt.hist(IDOA_values, label='Unhealthy', color='b', alpha = 0.5)


IDOA_values = []

for i in range(len(healthy_cohort)):
    healty = np.delete(healthy_cohort, i, axis=0)
    cohorts_list = [healty]

    models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
    data = {
        'models': models
    }
    idoa = IDOA.IDOA(data)
    IDOA_values.append(idoa.calculate_IDOA(healty, healthy_cohort[i]))

plt.hist(IDOA_values, label='Healthy', color='r', alpha = 0.5)

plt.legend()
plt.show()