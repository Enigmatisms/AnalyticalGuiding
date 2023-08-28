import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    origin = np.float32([0.523907, 3.909169, 2.090282, 9.757042, 9.484419, 18.283212])
    darts = np.float32([0.006528, 0.057450, 0.055036, 0.058269, 0.139562, 0.162919])
    origin = np.log1p(origin)
    darts = np.log1p(darts)
    ts = np.linspace(0.1, 0.6, 6)
    plt.scatter(ts, origin, color = 'red', s = 5)
    plt.scatter(ts, darts, color = 'blue', s = 5)
    plt.plot(ts, origin, color = 'red', label = 'variance for original method')
    plt.plot(ts, darts, color = 'blue', label = 'variance for DARTS')
    plt.legend()
    plt.grid(axis = 'both')
    plt.xlabel('scattering coefficient')
    plt.ylabel('variance (log(1 + x))')
    plt.show()