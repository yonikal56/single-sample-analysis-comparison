from modules import *
import matplotlib.pyplot as plt
import numpy as np

# set constants
m = 100  # num of samples per cohort
cohorts = 1
bound = 0.025
probability = 0.1

# create two different GLV models with m samples
file_path = '../schematic_idoa_samples.json'
data = GLV.generate_models(m, cohorts, file_path, bound=bound, probability=probability, force=True)
idoa = IDOA.IDOA(data)
null_model = GLV.GLV(r=data['models'][0]['r'], A=np.diag([-1] * GLV.GLV.numOfPopulations))
sample = null_model.get_sample()
samples, real = GLV.generate_random_shuffled_samples(data['models'][0], 1)
sample2 = samples[0]
samples, real = GLV.generate_random_samples(data, 1)
sample3 = samples[0]

cohort1 = data['models'][0]['cohort']

x1, y1 = idoa.calculate_IDOA(cohort1, sample, True)
x2, y2 = idoa.calculate_IDOA(cohort1, sample2, True)
x3, y3 = idoa.calculate_IDOA(cohort1, sample3, True)

x1 = np.array(x1)
y1 = np.array(y1)
x2 = np.array(x2)
y2 = np.array(y2)
x3 = np.array(x3)
y3 = np.array(y3)

a1, b1 = np.polyfit(x1, y1, 1)
plt.plot(x1, a1 * x1 + b1, label=f'null - {a1}x{b1}')
plt.scatter(x1, y1, s=5)
a2, b2 = np.polyfit(x2, y2, 1)
plt.plot(x2, a2 * x2 + b2, label=f'shuffled - {a2}x{b2}')
plt.scatter(x2, y2, s=5)
a3, b3 = np.polyfit(x3, y3, 1)
plt.plot(x3, a3 * x3 + b3, label=f'real - {a3}x{b3}')
plt.scatter(x3, y3, s=5)

plt.xlabel("Overlap", fontsize=13)
plt.ylabel("Dissimilarity", fontsize=13)
plt.legend()

plt.show()
