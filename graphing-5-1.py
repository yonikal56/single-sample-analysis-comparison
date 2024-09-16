from modules import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Qt5Agg")

# set constants
m = 10  # num of samples per cohort
cohorts = 2
GLV.GLV.numOfPopulations = 20

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(m, 1, file_path, force=True, sameR=False, bound=0.025)

fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [10, 1, 1], 'wspace': 0.5}, figsize =(11, 8))

samples, real = GLV.generate_random_shuffled_samples(data['models'][0], 1)

data['models'][0]['cohort'] = [[a / sum(sample) for a in sample] for sample in data['models'][0]['cohort']]
cohort = data['models'][0]['cohort']
samples = [[a / sum(sample) for a in sample] for sample in samples]


all_data = []
for i in range(m):
    all_data.append([f'{i}'] + cohort[i])

# create data
df1 = pd.DataFrame(all_data,
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df_real = pd.DataFrame([[0] + cohort[np.random.randint(m)]],
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
plt.show()

