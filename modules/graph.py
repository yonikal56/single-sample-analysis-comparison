import matplotlib.pyplot as plt
import numpy as np


class Graph:
    @staticmethod
    def scatter(x, y, axs, fit=True, set_y_lim=True):
        x = np.array(x)
        y = np.array(y)
        axs.scatter(x, y, s=1)
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        if fit:
            a, b = np.polyfit(x, y, 1)
            axs.plot(x, a * x + b, color='red', linestyle='-', linewidth=2)

    def plot(self, x, y, axs, color='red', line_style='-', line_width=2, set_y_lim=True):
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        axs.plot(x, y, color=color, linestyle=line_style, linewidth=line_width)

    def hist(self, x, axs):
        axs.hist(x, alpha=0.5)

    def legend(self):
        plt.legend()

    def boxplot(self, data, axs, positions, width):
        axs.boxplot(data, showfliers=False, positions=positions, widths=width)

    def get_axes(self):
        return self.__axs

    @staticmethod
    def show():
        plt.show()

    def __init__(self, *size):
        fig, self.__axs = plt.subplots(*size)
