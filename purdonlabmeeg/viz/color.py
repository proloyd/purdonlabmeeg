from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt


def _get_cycler(n=10):
    color = plt.cm.viridis(np.linspace(0, 1, n))
    return cycler('color', color)
