"""
    Multiple Importance Sampling test for the proposed directional sampling
    given the target position and the starting position (0, 0, 0)
    if we set T and g, we can have a scattering ellipse to integrate
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray as Arr
from scipy.spatial.transform import Rotation as Rot

from curve_viz import ellipse_t
from elliptical_phase import phase_hg, inverse_cdf_sample, eps_pdf

def np_rotation_between(fixd: Arr, target: Arr) -> Arr:
    """
        Transform parsed from xml file is merely camera orientation (numpy CPU version)
        INPUT arrays [MUST] be normalized
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = np.cross(fixd, target)
    dot = np.dot(fixd, target)
    if abs(dot) > 1. - 1e-5:            # nearly parallel
        return np.sign(dot) * np.eye(3, dtype = np.float32)  
    else:
        # Not in-line, cross product is valid
        axis /= np.linalg.norm(axis)
        axis *= np.arccos(dot)
        return Rot.from_rotvec(axis).as_matrix().astype(np.float32)
    
def delocalize_rotate(anchor: Arr, local_dir: Arr):
    R = np_rotation_between(np.float32([0, 1, 0]), anchor)
    return R @ local_dir

def sample_hg(g: float, num_samples = 100000):
    """ H-G sphere sampling: returns sampled direction and cos_theta """
    cos_theta = 0.
    g2 = g * g
    sqr_term = (1. - g2) / (1. + g - 2. * g * np.random.rand(num_samples, dtype = np.float32))
    cos_theta = (1. + g2 - sqr_term * sqr_term) / (2. * g)
    sin_theta = np.sqrt(np.maximum(0., 1. - cos_theta * cos_theta))
    phi = 2. * np.pi * np.random.rand(num_samples, dtype = np.float32)
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    return np.stack([np.cos(phi) * sin_theta, cos_theta, np.sin(phi) * sin_theta], axis = -1), cos_theta

def mc_local_sampling(g: float, T: float, input_dir: Arr, target_pos: Arr, num_samples = 100000):
    """ Monte Carlo local sampling method
        The final output will be integrated value and histogram
    """
    local_ray_dir, cos_theta = sample_hg(g, num_samples)
    # the local ray_dir should be transformed to the correct frame
    # The first scattering is sampled, so the PDF and phase function evaluation cancel each other out
    # but we still need it for MIS purpose
    pdf = phase_hg(g, cos_theta) / (2. * np.pi)
    ray_dir = delocalize_rotate(input_dir, local_ray_dir)
    # Do not use cos_theta here, the above cosine theta is defined against input_dir not x-axis. Use ray_dir instead
    ray_lens = ellipse_t(T, d, ray_dir[:, 0])
    ell_pos  = ray_lens * ray_dir
    new_dir  = target_pos - ell_pos
    new_dir /= np.linalg.norm(new_dir, axis = -1, keepdims = True)      # normalization
    cos_theta = (ray_dir * new_dir).sum(axis = -1)
    all_phase = phase_hg(g, cos_theta)
    to_emitter = T - ray_lens
    to_emitter *= to_emitter
    samples = all_phase / to_emitter            # phase function / d^2
    return samples, ray_dir, pdf

def get_ellipse_proba(g: float, d: float, T: float, alpha : float):
    r = T / d
    abs_g = abs(g)
    return abs_g / (abs_g + r * alpha)

def mis_ellipse_sampling(
    g: float, T: float, d: float, input_dir: Arr, 
    target_pos: Arr, alpha = 0.5, num_samples = 100000):
    """ MIS EPS method """
    # inverse_cdf_sample
    eta = get_ellipse_proba(g, d, T, alpha)
    mis_samples = np.random.rand(num_samples)
    ell_sample_cnt = (mis_samples > eta).sum()
    ori_sample_cnt = num_samples - ell_sample_cnt
    
    ori_results, ori_1st_rayd, ori_pdf = mc_local_sampling(g, T, input_dir, target_pos, ori_sample_cnt)
    ell_results, ell_1st_cos, ell_pdf = inverse_cdf_sample(g, d, T, ell_sample_cnt, True)
    # first, convert the ellipse 1st cosine term to ray direction, remember this cosine is related to the target direction
    phi = 2. * np.pi * np.random.rand(num_samples, dtype = np.float32)
    sin_theta = np.sqrt(np.maximum(0., 1. - ell_1st_cos * ell_1st_cos))
    ell_local_rayd = np.stack([np.cos(phi) * sin_theta, ell_1st_cos, np.sin(phi) * sin_theta], axis = -1)
    ell_rayd = delocalize_rotate(np.float32([1, 0, 0]), ell_local_rayd)
    cos_input_dir = (ell_rayd * input_dir).sum(axis = -1)
    # TODO: the 2 pi problem is not solved
    pdf_ell2ori = phase_hg(g, cos_input_dir) / (2. * np.pi)
    # then we need to calculate the PDF of using EPS to sample the original samples - `pdf_ori2ell`
    cos_x_ori = ori_1st_rayd[:, 0]
    pdf_ori2ell = eps_pdf(g, d, T, cos_x_ori)
    # ok we have almost everything we need, we should further (1) get 1 / d^2, (2) the first sampling and PDF does not cancel out
    d_emitter = T - ellipse_t(T, d, ell_rayd[:, 0])
    d_emitter *= d_emitter
    
    # now it is time to calculate MIS!
    mis_ori_samples = ori_results / (eta * pdf_ell2ori + (1 - eta) * ori_pdf)

    pass

if __name__ == "__main__":
    T = 4
    d = 1
    target_pos = np.float32([d, 0, 0])
    

