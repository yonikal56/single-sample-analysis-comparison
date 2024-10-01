import matplotlib.pyplot as plt
import matplotlib.image as mpimg


plt.rcParams["figure.figsize"] = [18, 15]
plt.rcParams["figure.autolayout"] = True

f, axes = plt.subplot_mosaic("AABB;CCDD", constrained_layout=True, gridspec_kw={

})

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
ax1 = axes['A']
ax2 = axes['B']
ax3 = axes['C']
ax4 = axes['D']
ax1.imshow(mpimg.imread("../article figures/5-setup.png"))
ax1.set_title('a', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
ax1.axis('off')
ax2.imshow(mpimg.imread("../article figures/5-hist.png"))
ax2.set_title('b', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
ax2.axis('off')
ax3.imshow(mpimg.imread("../article figures/5-idoa-wd2.png"))
ax3.set_title('c', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
ax3.axis('off')
ax4.imshow(mpimg.imread("../article figures/5-auc.png"))
ax4.set_title('d', loc='left', fontsize=15, weight='bold', x=0, y=0.9)
ax4.axis('off')

plt.savefig('../article figures/5-all.png')
