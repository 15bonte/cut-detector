""" Some plotting facilities
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_frame(frame: np.array, title: str):
    # vmin and vmax ?
    plt.imshow(frame, cmap="gray", vmin=np.min(frame), vmax=np.max(frame))
    plt.title(title)
    plt.show()