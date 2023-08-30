""" Visualize oval visualizer output
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from visualize_oval import evaluate_phase

def cdf_1(g: float, k1: float, samples: np.ndarray):
    """ Evaluate the first part, without normalization"""
    a = 2 * g * k1
    first_term = (1 - g ** 2) / a / np.sqrt(1 + g ** 2 - 2 * g * (-1 + k1 + k1 * samples))
    second_term = (1 - g) / a
    return first_term - second_term

def cdf_2_raw(g: float, k2: float, samples: np.ndarray):
    """ Evaluate the second part, without normalization and integral debias hence 'raw'"""
    a = 2 * g * k2
    first_term = (1 - g ** 2) / a / np.sqrt(1 + g ** 2 + 2 * g * (1 + k2 - k2 * samples))
    return first_term

def get_c2_z(g: float, k1: float, k2: float, cos_v: float):
    """ get integral constant for the second part of the piece-wise function and the normalization constant """
    a = 2 * g * k2
    first_term  = cdf_1(g, k1, cos_v)
    second_term = cdf_2_raw(g, k2, cos_v)
    c2 = first_term - second_term
    z = c2 + (1 - g) / a
    return c2, z

def phase_hg(g: float, samples: np.ndarray):
    """ Evaluate Henyey-Greenstein phase function """
    return (1 - g ** 2) / 2 / (1 + g ** 2 - 2 * g * samples) ** (3/2)

def get_k(d, T, is_first = True):
    """ The the slope for the first or second segment of cosine slope """
    dT2 = (d / T) ** 2
    if is_first:
        return 2 * dT2 / (d / T + 1)
    return -2 * dT2 / (-d / T + 1)

def inverse_map1(g: float, k1: float, Z: float, samples: np.ndarray):
    """ Inverse CDF sampling mapping function 1 """
    f2 = (1 - g**2) / (2 * g * k1 * Z * samples + (1 - g))
    f2 = f2 * f2
    nominator = 1 + g ** 2 + 2 * g * (1 - k1) - f2
    return nominator / (2 * g * k1)

def inverse_map2(g: float, k2: float, Z: float, C2: float, samples: np.ndarray):
    """ Inverse CDF sampling mapping function 2 """
    f2 = (1 - g**2) / (2 * g * k2 * (Z * samples - C2))
    f2 = f2 * f2
    nominator = 1 + g ** 2 + 2 * g * (1 + k2) - f2
    return nominator / (2 * g * k2)

def eval_second_scat(g: float, d: float, T: float, cos_samples: np.float32):
    x_scatter =  0.5 * (T + d) * (T - d) / (T - cos_samples * d)
    cos_2 = (x_scatter - d * cos_samples) / (x_scatter - T)
    return phase_hg(g, cos_2)

def inverse_cdf_sample(g: float, d: float, T: float, num_samples = 1000000):
    """ Inverse CDF sampling for the proposed direction sampling """
    NUM_BINS = 200
    rd_samples = np.random.rand(num_samples)
    k1 = get_k(d, T)
    k2 = get_k(d, T, False)
    cos_dT = d / T
    c2, z = get_c2_z(g, k1, k2, cos_dT)
    p1 = cdf_1(g, k1, cos_dT) / z
    is_part_1 = rd_samples < p1
    part_1_samps = inverse_map1(g, k1, Z = z, samples = rd_samples[is_part_1])
    part_2_samps = inverse_map2(g, k2, Z = z, C2 = c2, samples = rd_samples[~is_part_1])
    samples = np.concatenate([part_1_samps, part_2_samps])
    
    cos_thetas = np.linspace(-1 + 1e-5, 1, 2000)
    origin_phase2 = eval_second_scat(g, d, T, cos_thetas)
    hist, _ = np.histogram(samples, range = (-1.0, 1.0), bins = NUM_BINS)
    origin_phase2 *= hist.max() / origin_phase2.max()
    
    phase2_df = pd.DataFrame.from_dict({'cos':cos_thetas, 'val':origin_phase2})
    ax = sns.histplot(samples, binrange = (-1.0, 1.0), bins = NUM_BINS, alpha = 0.4, 
                 log_scale = (False, False), kde = True, label = 'Our sampling method', 
                 element = 'poly', line_kws={'label': "KDE fit", 'linestyle': '--'})
    ax.lines[0].set_color('#C63D2F')
    sns.lineplot(data = phase2_df, x = 'cos', y = 'val', label = 'Actual elliptical distribution', color = '#609966')

    scattering_prop = "Backward scattering"
    if g > 0:
        scattering_prop = "Forward scattering"
    if abs(g) >= 0.5:
        scattering_prop = f"{scattering_prop} (strong) g = "
    elif abs(g) >= 0.2:
        scattering_prop = f"{scattering_prop} (moderate) g = "
    else:
        scattering_prop = f"{scattering_prop} (weak) g = "
    plt.title(f"{scattering_prop}{g}, T (2a) = {T}, d (2c) = {d}")
    plt.grid(axis = 'both')
    plt.tight_layout()
    plt.legend()
    plt.show()

def visualize_curves(g = -0.5, d = 3, T = 6, num_samples = 1000, show_quadrature = False):
    r = d / T
    samples = np.linspace(-1, 1, num_samples)
    samples_1 = samples[samples < r]
    samples_2 = samples[samples >= r]
    k1 = get_k(d, T)
    k2 = get_k(d, T, False)
    
    if show_quadrature:
        cos_1 = k1 * (samples_1 + 1) - 1
        cos_2 = k2 * (samples_2 - 1) - 1
        cdf_v1 = phase_hg(g, cos_1)
        cdf_v2 = phase_hg(g, cos_2)
        cdf_v1 = np.cumsum(cdf_v1)
        cdf_v2 = np.cumsum(cdf_v2) + cdf_v1[-1]
        plt.legend("Numerical quadrature visualization")
    else:
        c2, z = get_c2_z(g, k1, k2, r)
        cdf_v1 = cdf_1(g, k1, samples_1) / z
        cdf_v2 = (cdf_2_raw(g, k2, samples_2) + c2) / z
        plt.legend("Analytical visualization")
    plt.plot(samples_1, cdf_v1, c = 'r', label = 'CDF part 1')
    plt.plot(samples_2, cdf_v2, c = 'b', label = 'CDF part 2')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TKAgg')
    d = 3
    T = 6
    g = 0.2
    # visualize_curves(g, d, T)
    inverse_cdf_sample(g, d, T)