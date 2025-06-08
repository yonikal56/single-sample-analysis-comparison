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

IDOA.IDOA.real = True
GLV.GLV.numOfPopulations = len(healthy_cohort[0])

fig, axs = plt.subplots(3, 2)
axs[0][0].set_title('healthy')
axs[0][1].set_title('unhealthy')

for row_num in range(3):
    healty_index = random.choice(range(len(healthy_cohort)))
    unhealty_index = random.choice(range(len(other_cohort)))

    healty = np.delete(healthy_cohort, healty_index, axis=0)
    cohorts_list = [healty]

    models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
    data = {
        'models': models
    }
    idoa = IDOA.IDOA(data)

    x_healthy, y_healthy = idoa.calculate_IDOA(healty, healthy_cohort[healty_index], True)
    print(x_healthy, y_healthy)
    poly_healthy = np.poly1d(np.polyfit(x_healthy, y_healthy, 1))
    print(poly_healthy)
    axs[row_num][0].scatter(x_healthy, y_healthy)
    axs[row_num][0].plot(x_healthy, poly_healthy(x_healthy), label=f'{poly_healthy}')
    axs[row_num][0].legend()

    x_unhealthy, y_unhealthy = idoa.calculate_IDOA(healty, other_cohort[unhealty_index], True)
    poly_unhealthy = np.poly1d(np.polyfit(x_unhealthy, y_unhealthy, 1))
    axs[row_num][1].scatter(x_unhealthy, y_unhealthy)
    axs[row_num][1].plot(x_unhealthy, poly_unhealthy(x_unhealthy), label=f'{poly_unhealthy}')
    axs[row_num][1].legend()

plt.legend()
plt.show()
