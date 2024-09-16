import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams["figure.figsize"] = [8, 13]
plt.rcParams["figure.autolayout"] = True

fig, axes = plt.subplot_mosaic("AA;BB;CE;DE", constrained_layout=True, gridspec_kw={
    'height_ratios': [3, 2, 1, 1],
    'hspace': 0.1
})

# a - cohorts + test sample
axes['A'].imshow(mpimg.imread("../article figures/1-1.png"), aspect='auto')
axes['A'].set_title('a', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
axes['A'].axis('off')

# b - distances
axes['B'].imshow(mpimg.imread("../article figures/1-2.png"), aspect='auto')
axes['B'].set_title('b', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
axes['B'].axis('off')

# c - NN
axes['C'].imshow(mpimg.imread("../article figures/1-3-2.png"), aspect='auto')
axes['C'].set_title('c', loc='left', fontsize=15, weight='bold', x=0, y=1.2)
axes['C'].set_xlabel(r'Neural Network (NN)', fontsize=10, y=2)
axes['C'].set_frame_on(False)
axes['C'].set_xticks([])
axes['C'].set_yticks([])
axes['C'].xaxis.set_label_position('top')

# d - IDOA
axes['D'].imshow(mpimg.imread("../article figures/1-3-1.png"))
axes['D'].set_title('e', loc='left', fontsize=15, weight='bold', x=-0.75, y=1.2)
axes['D'].set_xlabel(r'Individual Dissimilarity-Overlap Analysis (IDOA)', fontsize=10)
axes['D'].set_frame_on(False)
axes['D'].set_xticks([])
axes['D'].set_yticks([])
axes['D'].xaxis.set_label_position('top')

# e - NI
axes['E'].imshow(mpimg.imread("../article figures/1-3-3.png"), aspect='auto')
axes['E'].set_title('d', loc='left', fontsize=15, weight='bold', y=1.08)
axes['E'].set_xlabel(r'Network Impact (NI)', fontsize=10)
axes['E'].set_frame_on(False)
axes['E'].set_xticks([])
axes['E'].set_yticks([])
axes['E'].xaxis.set_label_position('top')

plt.savefig('../article figures/1-all.png')
