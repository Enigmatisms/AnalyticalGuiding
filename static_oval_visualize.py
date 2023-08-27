""" Visualize oval visualizer output
"""

import os
import sys
import natsort
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prefix = sys.argv[1]
    files = []
    for file in os.listdir("."):
        if not file.endswith(".npy"): continue
        if not file.startswith(prefix): continue
        files.append(file)
    files = natsort.natsorted(files)
    for file in files:
        no_ext = file[:-4]
        items = no_ext.split("_")
        data = np.fromfile(file, dtype = np.float32).reshape(2, -1)
        plt.plot(data[0], data[1], label = f'ratio = {items[-1]}')
    plt.xlabel('cosine theta value')
    plt.ylabel('cos alpha value')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()