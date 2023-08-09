import numpy as np
import matplotlib.pyplot as plt
from utils.colors import COLORS

def curve_calc(ts: np.ndarray, T, d, cos_t):
    """ dT / dt derivation """
    return (ts - d * cos_t) / (T - ts) + 1

def curve_calc_T(Ts: np.ndarray, d, cos_t):
    return 2 * ((Ts - d * cos_t) ** 2) / (Ts ** 2 - 2 * Ts * d * cos_t + d**2)

def ellipse_t(T, d, cos_t):
    return (T * T - d * d) / (T - d * cos_t) * 0.5

def get_ts(d, T, num_samples = 2000):
    return np.linspace(T - d, T + d, num_samples) * 0.5

def eval_Ts():
    plt.title("Different cos theta values (sample direction)")
    d = 4
    cos_ts = np.linspace(-1, 1, 7)
    Ts = np.linspace(4 + 1e-4, 25)
    for i, cos_t in enumerate(cos_ts):
        ts = ellipse_t(Ts, d, cos_t)
        curve = curve_calc(ts, Ts, d, cos_t)
        plt.plot(Ts, curve, label = f'cos theta = {cos_t:.3f}', c = COLORS[i])
    plt.xlabel("Total path length")
    plt.ylabel("Jacobian value")
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

def eval_ds():
    plt.title("Different cos theta values (sample direction)")
    ds = np.linspace(0.5, 19.9, 2000)
    cos_ts = np.linspace(-1, 1, 7)
    T = 20
    for i, cos_t in enumerate(cos_ts):
        ts = ellipse_t(T, ds, cos_t)
        curve = curve_calc(ts, T, ds, cos_t)
        plt.plot(ds, curve, label = f'cos theta = {cos_t:.3f}', c = COLORS[i])
    plt.xlabel("Path length to the target vertex")
    plt.ylabel("Jacobian value")
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    d = 4
    all_cos_ts = [[0.5, np.sqrt(2) / 2, np.sqrt(3) / 2]]
    all_cos_ts.append([-x for x in all_cos_ts[0]])
    Ts = np.linspace(4 + 1e-4, 25)
    title = ['positive cosine values', 'negative cosine values']
    for i, cos_ts in enumerate(all_cos_ts):
        plt.subplot(2, 1, i + 1)
        plt.title(f"{title[i]}")
        for j, cos_t in enumerate(cos_ts):
            ts = ellipse_t(Ts, d, cos_t)
            curve_1 = curve_calc(ts, Ts, d, cos_t)
            curve_2 = curve_calc_T(Ts, d, cos_t)
            plt.plot(Ts, curve_2, label = f'old derivation, cos_theta = {cos_t:.4f}', c = COLORS[2 * j + 1])
            plt.plot(Ts, curve_1, label = f'new derivation, cos_theta = {cos_t:.4f}', c = COLORS[2 * j], linestyle = '--')
            plt.xlabel("Total path length")
            plt.ylabel("Jacobian value")
            plt.grid(axis = 'both')
            plt.legend()
    plt.tight_layout()
    plt.show()
    