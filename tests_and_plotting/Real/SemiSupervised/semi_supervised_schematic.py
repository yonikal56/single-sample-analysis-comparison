from modules import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
#matplotlib.use("Qt5Agg")


# Load the CSV file
data_set = f'.'
file_path = f'{data_set}/for_preprocess.csv'
metadata_path = f'{data_set}/metadata.csv'
df = pd.read_csv(file_path)
meta_data_df = pd.read_csv(metadata_path)
samples = {}
labels = {}


healthy_count = 0
sick_count = 0
for index, row in meta_data_df.iterrows():
    labels[row['ID']] = row['Tag']  # 0 for healthy, 1 for sick
    if row['Tag'] == 0:
        healthy_count += 1
    elif row['Tag'] == 1:
        sick_count += 1


for index, row in df.iterrows():
    list = row.tolist()
    if (list[0].startswith('SRR')):
        samples[list[0]] = list[1:]
        samples[list[0]] = [float(a) for a in samples[list[0]]]


hit_numbers = []
species = []
keys_to_remove = []
for key, sample in samples.items():
    hit_numbers.append(sum(sample))
    species.append(sum([(1 if a != 0 else 0) for a in sample]))


for key in keys_to_remove:
    del samples[key]
    del labels[key]

healthy_cohort = []
for key, label in labels.items():
    if key not in samples.keys():
        continue
    if label == 0:  # healthy
        healthy_cohort.append(np.array(samples[key])/sum(samples[key]))


IDOA.IDOA.real = True
GLV.GLV.numOfPopulations = len(healthy_cohort[0])
m = 10


fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [10, 1, 1], 'wspace': 0.5}, figsize =(11, 8))

samples = healthy_cohort.copy()
# each index is shuffled sample is a value from random sample inside cohort in the same index
samples = [np.array([random.choice(samples[:m])[i] for i in range(GLV.GLV.numOfPopulations)])]
samples = [[a / sum(sample) for a in sample] for sample in samples]

cohort = healthy_cohort

all_data = []
for i in range(m):
    all_data.append([f'{i}'] + cohort[i].tolist())


# create data
df1 = pd.DataFrame(all_data,
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df_real = pd.DataFrame([[0] + cohort[np.random.randint(m)].tolist()],
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df_shuffled = pd.DataFrame([np.append(np.array([0]), samples[0])],
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))

# plot data in stack manner of bar type
ax1 = df1.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[0], title='Cohort')
ax2 = df_real.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[1], title='Real sample')
ax3 = df_shuffled.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[2], title='Shuffled sample')
ax1.get_legend().remove()
ax2.get_legend().remove()
ax3.get_legend().remove()
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.title.set_size(13)
ax2.title.set_size(13)
ax3.title.set_size(13)
plt.savefig('../../../../article figures/5-real-setup.png')
plt.show()

