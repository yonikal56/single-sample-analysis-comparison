import matplotlib.pyplot as plt
import numpy as np


class Graph:
    @staticmethod
    def scatter(x, y, axs, fit=False, set_y_lim=False, color=None, size=1, label=None):
        x = np.array(x)
        y = np.array(y)
        axs.scatter(x, y, color=color, s=size, label=label)
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        if fit:
            a, b = np.polyfit(x, y, 1)
            axs.plot(x, a * x + b, color='red', linestyle='-', linewidth=2)

    def plot(self, x, y, axs, color='red', line_style='-', line_width=2, set_y_lim=True, label=None):
        if set_y_lim:
            axs.set_ylim(0, max(y) + 0.05)
        axs.plot(x, y, color=color, linestyle=line_style, linewidth=line_width, label=label)

    def hist(self, x, axs, color=None, alpha=0.5, label=None):
        axs.hist(x, alpha=alpha, ec=color, density=True, label=label)

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

    def set_title(self, title):
        self.__fig.suptitle(title, fontsize=16)

    def __init__(self, *size, title=None):
        self.__fig, self.__axs = plt.subplots(*size)
        if title is not None:
            self.__fig.canvas.manager.set_window_title(title)
