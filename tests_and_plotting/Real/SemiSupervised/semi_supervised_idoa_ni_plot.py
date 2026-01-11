import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import random
from modules import *
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

# set constants

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
        healthy_cohort.append(np.array(samples[key]))


IDOA.IDOA.real = True
GLV.GLV.numOfPopulations = len(healthy_cohort[0])

m = len(healthy_cohort)
num_of_samples = int(m / 2)
shuffled_cohort = [np.array([random.choice(healthy_cohort)[i] for i in range(GLV.GLV.numOfPopulations)]) for _ in range(num_of_samples)]

data = {
    'models': [{
        'cohort': healthy_cohort[:100],
    },
    {
        'cohort': shuffled_cohort
    }]
}
data['models'][0]['cohort'] = [sample/sum(sample) for sample in data['models'][0]['cohort']]
data['models'][1]['cohort'] = [sample/sum(sample) for sample in data['models'][1]['cohort']]
samples = healthy_cohort[100:]
shuffled_samples = [np.array([random.choice(samples)[i] for i in range(GLV.GLV.numOfPopulations)]) for _ in range(40)]
samples = samples + shuffled_samples
samples = [sample/sum(sample) for sample in samples]
real = [0] * 40 + [1] * 40


idoa = IDOA.IDOA(data)
network_impact = NetworkImpact.NetworkImpact(data)

# predictions
network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)

idoa_predictions = idoa.predict_real(data['models'][0]['cohort'], np.array(samples))
ni_wd2 = network_impact3.predict_real(data['models'][0]['cohort'], np.array(samples))

x = []
y = []
x2 = []
y2 = []

for i in range(len(idoa_predictions)):
    if real[i] == 0:
        x.append(idoa_predictions[i])
        y.append(ni_wd2[i])
    else:
        x2.append(idoa_predictions[i])
        y2.append(ni_wd2[i])

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111)
ax.scatter(x, y, label="Real")
ax.scatter(x2, y2, label="Shuffled")
ax.ticklabel_format(axis="x", style="sci", scilimits=(1,2))
ax.ticklabel_format(axis="y", style="sci", scilimits=(1,2))
ax.set_xlabel('IDOA', fontsize=13)
ax.set_ylabel('NI - WD2', fontsize=13)
plt.legend()
plt.savefig('../../../../article figures/5-real-idoa-wd2.png')
plt.show()
