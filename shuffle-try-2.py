from modules import *
import numpy as np
import matplotlib.pyplot as plt

# set constants
m = 100
num_of_samples = 200

file_path = 'shuffled-samples.json'
data = GLV.generate_models(m, 1, file_path, force=True, bound=0.025)

samples, real = GLV.generate_random_shuffled_samples(data['models'][0], num_of_samples)
data['models'][0]['cohort'] = [[a / sum(sample) for a in sample] for sample in data['models'][0]['cohort']]
samples = [[a / sum(sample) for a in sample] for sample in samples]

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
plt.show()
