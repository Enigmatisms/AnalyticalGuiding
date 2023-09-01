""" Visualize oval visualizer output
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cdf_1(g: float, k1: float, samples: np.ndarray):
    """ Evaluate the first part, without normalization"""
    a = 2 * g * k1
    first_term = (1 - g ** 2) / a / np.sqrt(1 + g ** 2 - 2 * g * (-1 + k1 + k1 * samples))
    second_term = (1 - g) / a
    return first_term - second_term

def cdf_1_cu(g: float, k1: float, samples: np.ndarray):
    """ Evaluate the first part, without normalization"""
    a = 2 * g * k1
    first_term = (1 - g ** 2) / a / torch.sqrt(1 + g ** 2 - 2 * g * (-1 + k1 + k1 * samples))
    second_term = (1 - g) / a
    return first_term - second_term

def cdf_2_raw_cu(g: float, k2: float, samples: np.ndarray):
    """ Evaluate the second part, without normalization and integral debias hence 'raw'"""
    a = 2 * g * k2
    first_term = (1 - g ** 2) / a / torch.sqrt(1 + g ** 2 + 2 * g * (1 + k2 - k2 * samples))
    return first_term

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
    """ Evaluate Henyey-Greenstein phase function 
        Note that this function is not for spherical sampling but 2D ellipse sampling
        therefore the value for PDF should be divided by 2pi
    """
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

def eps_pdf(g: float, d: float, T: float, samples: float, cuda = False):
    """ input samples are cosine theta to the x-axis """
    k1 = get_k(d, T)
    k2 = get_k(d, T, False)
    cos_dT = d / T
    _, z = get_c2_z(g, k1, k2, cos_dT)
    if cuda:
        values = torch.zeros_like(samples, dtype = torch.float32).cuda()
    else:
        values = np.zeros_like(samples, dtype = np.float32)
    part_1_flag = samples < cos_dT
    values[part_1_flag]  = phase_hg(g, k1 * (samples[part_1_flag] + 1) - 1)
    values[~part_1_flag] = phase_hg(g, k2 * (samples[~part_1_flag] - 1) - 1)
    return values / (2. * np.pi * z)

def inverse_cdf_sample_cu(g: float, d: float, T: float, num_samples = 1000000, sample_only = False):
    """ Inverse CDF sampling for the proposed direction sampling """
    rd_samples = torch.rand(num_samples).cuda()
    k1 = get_k(d, T)
    k2 = get_k(d, T, False)
    cos_dT = d / T
    c2, z = get_c2_z(g, k1, k2, cos_dT)
    p1 = cdf_1(g, k1, cos_dT) / z
    is_part_1 = rd_samples < p1
    part_1_samps = inverse_map1(g, k1, Z = z, samples = rd_samples[is_part_1])
    part_2_samps = inverse_map2(g, k2, Z = z, C2 = c2, samples = rd_samples[~is_part_1])
    samples = torch.cat([part_1_samps, part_2_samps])
    if sample_only:
        # we should return sampling PDF and the evaluation result
        part1_eval = eval_second_scat(g, d, T, part_1_samps) / (2. * np.pi)
        part2_eval = eval_second_scat(g, d, T, part_2_samps) / (2. * np.pi)
        
        part1_pdf = phase_hg(g, k1 * (part_1_samps + 1) - 1) / (2. * np.pi) / z
        part2_pdf = phase_hg(g, k2 * (part_2_samps - 1) - 1) / (2. * np.pi) / z
        return torch.cat([part1_eval, part2_eval]), samples, torch.cat([part1_pdf, part2_pdf])

def inverse_cdf_sample(g: float, d: float, T: float, num_samples = 1000000, sample_only = False):
    """ Inverse CDF sampling for the proposed direction sampling """
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
    if sample_only:
        # we should return sampling PDF and the evaluation result
        part1_eval = eval_second_scat(g, d, T, part_1_samps) / (2. * np.pi)
        part2_eval = eval_second_scat(g, d, T, part_2_samps) / (2. * np.pi)
        
        part1_pdf = phase_hg(g, k1 * (part_1_samps + 1) - 1) / (2. * np.pi) / z
        part2_pdf = phase_hg(g, k2 * (part_2_samps - 1) - 1) / (2. * np.pi) / z
        return np.concatenate([part1_eval, part2_eval]), samples, np.concatenate([part1_pdf, part2_pdf])
    
    NUM_BINS = 200
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
    
def simulate_conversion(fix_angle, num_samples = 4000000, uniform_cosine = True):
    """ I am bother by this problem, and this is the last god damn step! 
        The purpose: try to uniformly sample the cos-alpha (related to x-axis)
        and calculate the cosine values related to a specific angle
    """
    BIN_NUMS = 1000
    if uniform_cosine:
        cos_values = np.random.rand(num_samples) * 2. - 1.
        sign_samples = np.random.rand(num_samples)
        angles = np.arccos(cos_values)
        angles[sign_samples < 0.5] *= -1
    else:
        angles = np.linspace(-np.pi + 1e-5, np.pi - 1e-5, num_samples)
    delta_angle = (fix_angle - angles) % (2. * np.pi) - np.pi
    new_cos = np.cos(delta_angle)
    plt.subplot(2, 1, 1)
    sns.histplot(new_cos, binrange = (-1.0, 1.0), bins = BIN_NUMS, alpha = 0.4, 
                 log_scale = (False, False), kde = False, label = 'Our sampling method', 
                 element = 'poly', line_kws={'label': "KDE fit", 'linestyle': '--'})
    plt.legend()
    plt.grid(axis = 'both')
    plt.subplot(2, 1, 2)
    alphas_1 = np.linspace(-np.pi + fix_angle + 0.03, fix_angle - 0.03, 2000)
    
    # note that alphas_2 is in reverse order
    alphas_2 = np.linspace(fix_angle + np.pi - 0.03, fix_angle + 0.03, 2000)
    xs_1 = np.cos(alphas_1 - fix_angle)
    plt.plot(xs_1, np.abs(np.sin(alphas_1) / np.sin(alphas_1 - fix_angle)) + np.abs(np.sin(alphas_2) / np.sin(alphas_2 - fix_angle)), c = 'r')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TKAgg')
    d = 1
    T = 4
    g = -0.6
    # # visualize_curves(g, d, T)
    inverse_cdf_sample(g, d, T)
    # fixed = 45 / 180 * np.pi
    # simulate_conversion(fixed, uniform_cosine = False)