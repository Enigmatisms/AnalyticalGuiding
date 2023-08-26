import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    origin = np.float32([0.789548, 1.649419, 2.122529, 4.945310, 5.064875, 26.616710])
    darts = np.float32([0.007415, 0.075409, 0.054035, 0.051419, 0.114190, 0.128300])
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