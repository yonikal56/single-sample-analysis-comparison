from matplotlib.patches import ConnectionPatch

from modules import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# set constants
m = 10  # num of samples per cohort
cohorts = 2
GLV.GLV.numOfPopulations = 20

# create two different GLV models with m samples
file_path = 'samples.json'
data = GLV.generate_models(m, 1, file_path, force=True, sameR=False, bound=0.025)

fig, axes = plt.subplot_mosaic("ABE;CCC", constrained_layout=True, gridspec_kw={
    'width_ratios': [10, 1, 1],
    'wspace': 0.2,
    'hspace': 0.3
})
axes['E'].axis('off')

cohort1 = data['models'][0]['cohort']
all_data1 = []
for i in range(m):
    all_data1.append([f'{i}'] + cohort1[i])

# create data
df1 = pd.DataFrame(all_data1,
                   columns=(['V'] + [f'{i}' for i in range(GLV.GLV.numOfPopulations)]))
df_one = pd.DataFrame([[0] + cohort1[np.random.randint(m)]],
                      columns=(['V'] + [f'{i}' for i in range(GLV.GLV.numOfPopulations)]))

# plot data in stack manner of bar type
ax1 = df1.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes['A'], title='Cohort A')
ax2 = df_one.plot(x='V', kind='bar', stacked=True, align='edge', width=1.0, ax=axes['B'], title='Test sample')
ax1.get_legend().remove()
ax2.get_legend().remove()
ax1.axis('off')
ax2.axis('off')
ax1.title.set_size(13)
ax2.title.set_size(13)
axes['C'].imshow(mpimg.imread("../article figures/1-3-network.png"), aspect='auto')
axes['C'].axis('off')
axes['C'].set_title('Ref.', fontsize=13, loc='left', x=0.2)
axes['C'].text(520, -6, 'Ref. + Test', size=13)

plt.show()
