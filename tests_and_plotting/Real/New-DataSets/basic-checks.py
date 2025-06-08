import pandas as pd
from modules import *
import numpy as np
import random
import matplotlib.pyplot as plt


# Load the CSV file
data_set = f'gc/gc1'
file_path = f'{data_set}/for_preprocess.csv'
metadata_path = f'{data_set}/metadata.csv'
df = pd.read_csv(file_path)
meta_data_df = pd.read_csv(metadata_path)
samples = {}
labels = {}
target_sum = 30000


def probabilistic_downsample(vector, target_sum=target_sum):
    vector = np.array(vector, dtype=float)
    if np.sum(vector) == 0:
        return np.zeros_like(vector, dtype=int)

    probabilities = vector / np.sum(vector)
    downsampled = np.random.multinomial(target_sum, probabilities)
    return downsampled

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
    if (sum(sample) < target_sum):
        keys_to_remove.append(key)
        continue
    sample = probabilistic_downsample(sample)
    samples[key] = sample
    hit_numbers.append(sum(sample))
    species.append(sum([(1 if a != 0 else 0) for a in sample]))

print(min(hit_numbers), max(hit_numbers))

for key in keys_to_remove:
    del samples[key]
    del labels[key]

healthy_cohort = []
other_cohort = []
for key, label in labels.items():
    if key not in samples.keys():
        continue
    if label == 0:  # healthy
        healthy_cohort.append(np.array(samples[key])/target_sum)
    elif label == 1:  # sick
        other_cohort.append(np.array(samples[key])/target_sum)

hit_numbers = []
species = []
for key, sample in samples.items():
    hit_numbers.append(sum(sample))
    species.append(sum([(1 if a != 0 else 0) for a in sample]))

# plt.hist(hit_numbers, alpha = 0.5, label=f'{len(samples)} samples, healthy={healthy_count}, sick={sick_count}')
# plt.hist(species, color='r', alpha = 0.5)
# plt.show()

print(len(healthy_cohort), len(other_cohort), len(healthy_cohort[0]))

IDOA.IDOA.real = True
GLV.GLV.numOfPopulations = len(healthy_cohort[0])

fig, axs = plt.subplots(2)
axs[0].set_title('healthy')
axs[1].set_title('unhealthy')

idoa_healthy = []
idoa_unhealthy = []

for i in range(len(healthy_cohort)):
    sample = np.array(healthy_cohort[i])
    healty = np.delete(healthy_cohort, i, axis=0)
    cohorts_list = [healty]

    models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
    data = {
        'models': models
    }
    idoa = IDOA.IDOA(data)

    idoa_healthy.append(idoa.calculate_IDOA(healty, sample))

for i in range(len(other_cohort)):
    sample = np.array(other_cohort[i])
    healty = healthy_cohort
    cohorts_list = [healty]

    models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
    data = {
        'models': models
    }
    idoa = IDOA.IDOA(data)

    idoa_unhealthy.append(idoa.calculate_IDOA(healty, sample))

axs[0].hist(idoa_healthy)
axs[1].hist(idoa_unhealthy)
axs[0].set_xlim([-0.6, 1.2])
axs[1].set_xlim([-0.6, 1.2])

plt.show()

doc = DOC.DOC()

overlaps = []
dissimilarities = []
for i in range(len(healthy_cohort)):
    for j in range(len(healthy_cohort)):
        if i == j:
            continue

        dissimilarity, overlap = doc.get_dissimilarity_overlap_point(healthy_cohort[i], healthy_cohort[j])
        overlaps.append(overlap)
        dissimilarities.append(dissimilarity)

poly_healthy = np.poly1d(np.polyfit(overlaps, dissimilarities, 1))
# print(poly_healthy)
# plt.scatter(overlaps, dissimilarities)
# plt.plot(overlaps, poly_healthy(overlaps), label=f'{poly_healthy}')
# plt.legend()
# plt.show()