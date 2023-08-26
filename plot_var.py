import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    origin = np.float32([0.400814, 1.129245, 1.593011, 3.675406, 2.367448, 25.984900])
    darts = np.float32([0.003807, 0.051048, 0.040090, 0.037951, 0.044579, 0.080963])
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