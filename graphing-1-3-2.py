from modules import *
import matplotlib.pyplot as plt
import numpy as np

# set constants
m = 100  # num of samples per cohort
cohorts = 2
bound = 0.025
probability = 0.1

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(m, cohorts, file_path, bound=bound, probability=probability, force=True)
idoa = IDOA.IDOA(data)
samples, real = GLV.generate_random_samples(data, 1)
sample = samples[0]

cohort1 = data['models'][0]['cohort']
cohort2 = data['models'][1]['cohort']

x1, y1 = idoa.calculate_IDOA(cohort1, sample, True)
x2, y2 = idoa.calculate_IDOA(cohort2, sample, True)

x1 = np.array(x1)
x2 = np.array(x2)
y1 = np.array(y1)
y2 = np.array(y2)

a, b = np.polyfit(x1, y1, 1)
a2, b2 = np.polyfit(x2, y2, 1)
plt.plot(x1, a * x1 + b, label=('Healthy' if real else 'Sick'))
plt.plot(x2, a2 * x2 + b2, label=('Healthy' if not real else 'Sick'))
plt.scatter(x1, y1, s=5)
plt.scatter(x2, y2, s=5)

plt.xlabel("Overlap", fontsize=13)
plt.ylabel("Dissimilarity", fontsize=13)
plt.xticks([])
plt.yticks([])
plt.legend()

plt.show()
