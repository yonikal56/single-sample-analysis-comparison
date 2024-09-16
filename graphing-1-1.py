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
data = GLV.generate_models(m, cohorts, file_path, force=True, sameR=True, bound=0.025)

fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [10, 1, 10]})

cohort1 = data['models'][0]['cohort']
cohort2 = data['models'][1]['cohort']
all_data1 = []
all_data2 = []
for i in range(m):
    all_data1.append([f'{i}'] + cohort1[i])
    all_data2.append([f'{i}'] + cohort2[i])

# create data
df1 = pd.DataFrame(all_data1,
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df2 = pd.DataFrame(all_data2,
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df_one = pd.DataFrame([[0] + cohort1[np.random.randint(m)]],
                  columns=(['V']+[f'{i}' for i in range(GLV.GLV.numOfPopulations)]))

# plot data in stack manner of bar type
ax1 = df1.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[0], title='Cohort A')
ax2 = df_one.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[1], title='Test sample')
ax3 = df2.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes[2], title='Cohort B')
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

