"""
    This is the CUDA version for faster numerical testing
    Multiple Importance Sampling test for the proposed directional sampling
    given the target position and the starting position (0, 0, 0)
    if we set T and g, we can have a scattering ellipse to integrate
"""

import time
import torch
import numpy as np

from torch import Tensor as Arr
from scipy.spatial.transform import Rotation as Rot

from curve_viz import ellipse_t
from elliptical_phase import phase_hg, inverse_cdf_sample_cu, eps_pdf

""" Let's make things faster """

def rotation_between(fixd: Arr, target: Arr) -> Arr:
    """
        Transform parsed from xml file is merely camera orientation (numpy CPU version)
        INPUT arrays [MUST] be normalized
        Orientation should be transformed to be camera rotation matrix
        Rotation from <fixed> vector to <target> vector, defined by cross product and angle-axis
    """
    axis = np.cross(fixd, target)
    dot = np.dot(fixd, target)
    if abs(dot) > 1. - 1e-5:            # nearly parallel
        R = np.sign(dot) * np.eye(3, dtype = np.float32)  
    else:
        # Not in-line, cross product is valid
        axis /= np.linalg.norm(axis)
        axis *= np.arccos(dot)
        R = Rot.from_rotvec(axis).as_matrix().astype(np.float32)
    return torch.from_numpy(R).cuda()
    
def delocalize_rotate(local_dir: Arr, R: Arr):
    if local_dir.ndim == 2:
        return (R @ local_dir[..., None]).squeeze()
    return R @ local_dir

def sample_hg(g: float, num_samples = 100000):
    """ H-G sphere sampling: returns sampled direction and cos_theta """
    cos_theta = 0.
    g2 = g * g
    sqr_term = (1. - g2) / (1. + g - 2. * g * torch.rand(num_samples, dtype = torch.float32)).cuda()
    cos_theta = (1. + g2 - sqr_term * sqr_term) / (2. * g)
    sin_theta = torch.sqrt((1. - cos_theta * cos_theta).clamp(0, 1))
    phi = 2. * np.pi * torch.rand(num_samples, dtype = torch.float32).cuda()
    # rotational offset w.r.t axis [0, 1, 0] & pdf
    return torch.stack([torch.cos(phi) * sin_theta, cos_theta, torch.sin(phi) * sin_theta], dim = -1), cos_theta

def mc_local_sampling(
    g: float, T: float, target_pos: Arr, 
    R: Arr, num_samples = 100000, mis = False):
    """ Monte Carlo local sampling method
        The final output will be integrated value and histogram
    """
    local_ray_dir, cos_theta = sample_hg(g, num_samples)
    # the local ray_dir should be transformed to the correct frame
    # The first scattering is sampled, so the PDF and phase function evaluation cancel each other out
    # but we still need it for MIS purpose
    pdf = phase_hg(g, cos_theta) / (2. * np.pi)
    ray_dir = delocalize_rotate(local_ray_dir, R)
    ray_dir /= torch.norm(ray_dir, dim = -1, keepdim = True)
    # Do not use cos_theta here, the above cosine theta is defined against input_dir not x-axis. Use ray_dir instead
    ray_lens = ellipse_t(T, d, ray_dir[:, 0])
    ell_pos  = ray_lens[..., None] * ray_dir
    new_dir  = -ell_pos + target_pos
    # open3d_plot(ell_pos, target_pos, input_dir)
    new_dir /= torch.norm(new_dir, dim = -1, keepdim = True)      # normalization
    cos_theta = (ray_dir * new_dir).sum(axis = -1)
    all_phase = phase_hg(g, cos_theta) / (2. * np.pi)
    # to_emitter = T - ray_lens
    # to_emitter *= to_emitter
    samples = all_phase # / to_emitter            # phase function / d^2
    if mis:
        samples *= pdf
    return samples, ray_dir, pdf

def get_ellipse_proba(g: float, d: float, T: float, alpha : float):
    r = T / d
    abs_g = abs(g)
    return abs_g / (abs_g + r * alpha)

def mis_ellipse_sampling(
    g: float, T: float, d: float, input_dir: Arr, target_pos: Arr, 
    alpha:float, R: Arr, num_samples = 100000, verbose = True):
    """ MIS EPS method """
    # inverse_cdf_sample
    eta = get_ellipse_proba(g, d, T, alpha)
    mis_samples = torch.rand(num_samples, dtype = torch.float32).cuda()
    ell_sample_cnt = (mis_samples < eta).sum()
    ori_sample_cnt = num_samples - ell_sample_cnt
    
    if verbose:
        print(f"Ellipse proba: {eta:.3f}. Actual ellipse samples: {ell_sample_cnt / num_samples * 100:.3f} %.")
    
    ori_results, ori_1st_rayd, ori_pdf = mc_local_sampling(g, T, target_pos, R, ori_sample_cnt, mis = True)
    ell_results, ell_1st_cos, ell_pdf = inverse_cdf_sample_cu(g, d, T, ell_sample_cnt, True)
    # first, convert the ellipse 1st cosine term to ray direction, remember this cosine is related to the target direction
    phi = 2. * np.pi * torch.rand(ell_sample_cnt, dtype = torch.float32).cuda()
    sin_theta = torch.sqrt((1. - ell_1st_cos * ell_1st_cos).clamp(0, 1))
    ell_local_rayd = torch.stack([torch.cos(phi) * sin_theta, ell_1st_cos, torch.sin(phi) * sin_theta], dim = -1)
    ell_rayd = delocalize_rotate(ell_local_rayd, R)
    cos_input_dir = (ell_rayd * input_dir).sum(axis = -1)
    # open3d_plot(ell_rayd, target_pos, input_dir)
    pdf_ell2ori = phase_hg(g, cos_input_dir) / (2. * np.pi)
    # then we need to calculate the PDF of using EPS to sample the original samples - `pdf_ori2ell`
    cos_x_ori = ori_1st_rayd[:, 0]
    pdf_ori2ell = eps_pdf(g, d, T, cos_x_ori, cuda = True)
    # ok we have almost everything we need, we should further (1) get 1 / d^2, (2) the first sampling and PDF does not cancel out
    # d_emitter = T - ellipse_t(T, d, ell_rayd[:, 0])
    # d_emitter *= d_emitter
    
    ell_results *= pdf_ell2ori # / d_emitter
    
    # now it is time to calculate MIS!
    # ori_samples: how we evaluate the PDF of original samples as if it is sampled by EPS
    mis_ori_samples = ori_results / (eta * pdf_ori2ell + (1 - eta) * ori_pdf)
    mis_ell_samples = ell_results / (eta * ell_pdf + (1 - eta) * pdf_ell2ori)
    
    all_samples = torch.cat([mis_ori_samples, mis_ell_samples])

    return all_samples.mean().item()
    
def variance_test(
    g: float, T: float, d: float, alphas: float, 
    input_dir: Arr, n_samples: int = 400000, num_iter: int = 4000
):
    import tqdm
    mc_samples = []
    mis_samples = []
    target_pos = np.float32([d, 0, 0])
    R_mc  = rotation_between(np.float32([0, 1, 0]), input_dir)
    R_mis = rotation_between(np.float32([0, 1, 0]), np.float32([1, 0, 0]))
    input_dir  = torch.from_numpy(input_dir).cuda()
    target_pos = torch.from_numpy(target_pos).cuda()
    for _ in tqdm.tqdm(range(num_iter)):
        samples, _, _ = mc_local_sampling(g, T, target_pos, R_mc, n_samples)
        mc_estimate = samples.mean().item()
        mis_estimate = mis_ellipse_sampling(g, T, d, input_dir, target_pos, alphas, R_mis, n_samples, verbose = False)
        mc_samples.append(mc_estimate)
        mis_samples.append(mis_estimate)
        
    mc_samples = np.float32(mc_samples)
    mis_samples = np.float32(mis_samples)
    mc_samples.tofile("mc_samples.npy")
    mis_samples.tofile("mis_samples.npy")
    
    all_mean = np.mean(np.concatenate([mc_samples, mis_samples]))
    var_mc = (mc_samples - all_mean).var()
    var_mis = (mis_samples - all_mean).var()
    print(f"Finished, the estimation is {all_mean:.5f}. Var ratio: MIS / MC = {var_mis / var_mc:.4f}")
    

if __name__ == "__main__":
    NUM_ITER = 4000
    N_SAMPLES = 1000000
    T = 1.5
    d = 1
    g = -0.7
    alphas = 0.5
    input_dir = np.float32([-1, 0, 1])
    input_dir /= np.linalg.norm(input_dir)
    
    variance_test(g, T, d, alphas, input_dir, N_SAMPLES, NUM_ITER)