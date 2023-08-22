import matplotlib.pyplot as plt
import numpy as np


class Graph:
    @staticmethod
    def scatter(x, y, axs, fit=True, set_y_lim=True, color=None):
        x = np.array(x)
        y = np.array(y)
        axs.scatter(x, y, s=1, color=color)
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        if fit:
            a, b = np.polyfit(x, y, 1)
            axs.plot(x, a * x + b, color='red', linestyle='-', linewidth=2)

    def plot(self, x, y, axs, color='red', line_style='-', line_width=2, set_y_lim=True, label=None):
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        axs.plot(x, y, color=color, linestyle=line_style, linewidth=line_width, label=label)

    def hist(self, x, axs, color=None, alpha=0.5):
        axs.hist(x, alpha=alpha, ec=color, density=True)

    def bar(self, x, y, axs, labels=None, colors=None):
        axs.barh(x,y, color=colors, label=labels)

    def legend(self, size=None):
        if size is not None:
            plt.legend(prop={'size': size})
        else:
            plt.legend()

    def boxplot(self, data, axs, positions, width):
        axs.boxplot(data, showfliers=False, positions=positions, widths=width)

    def get_axes(self):
        return self.__axs

    def get_fig(self):
        return self.__fig

    def get_plt(self):
        return plt

    @staticmethod
    def show():
        plt.show()

    def __init__(self, *size):
        self.__fig, self.__axs = plt.subplots(*size)
