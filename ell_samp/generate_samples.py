""" Generate training samples
    @date: 2023.3.16
    @author: Qianyue He
"""

import torch

PI_2 = torch.pi * 2

TENSOR_PROP = {"device": "cuda:0", "dtype": torch.float32}

def duplicate_inplace(x: torch.Tensor, dup: int = 8, interleave = False):
    """ Input x should be of shape (resolution, N, dim) """
    if interleave:
        resolution, _, channel = x.shape
        res = x[..., None, :].repeat(1, 1, dup, 1).view(resolution, -1, channel)
    else:
        res = x.repeat(1, dup, 1)
    return res

def merge_samples(x: torch.Tensor, y: torch.Tensor, dup = 8):
    x_n = duplicate_inplace(x, dup, interleave = True)
    dup_time = x_n.shape[1] // dup
    y_n = duplicate_inplace(y, dup_time, interleave = False)
    return torch.cat((x_n, y_n), dim = -1)

def generate_row(a_grid, a_div, r_s, r_e, resolution = 256, sp_per_dim = 8):
    """ FIXME: this sample generation needs to consider the grid line
        generate samples (each row)
        return is of shape (resolution, sp_per_dim ^ 3, 4), for normal setting\
        256 and 8, the GPU gmem consumption is 1.5 MB
        - a_grid: shape (resolition, 1)
        - a_div: float
        - r_s, r_e: ratio start and ratio end
    """
    # generate alpha, d/T samples (8, 2), note that d/T is truncated at 0.2
    rd_sample = torch.rand(resolution, sp_per_dim, 2, **TENSOR_PROP)           # (resolution, sp_per_dim, 2)
    rd_sample[..., 0] = rd_sample[..., 0] * a_div + a_grid            # uniform in (-pi, pi) -> alpha
    rd_sample[..., 1] = rd_sample[..., 1] * (r_e - r_s) + r_s         # uniform in (0.2, 1)  -> d/T
    # generate phi_theta_samples
    phi_sample = torch.rand(resolution, sp_per_dim, 1, **TENSOR_PROP) * PI_2 - torch.pi      # (-PI, PI)
    tht_sample = torch.rand(resolution, sp_per_dim, 1, **TENSOR_PROP) * torch.pi             # (0, PI)
    rd_sample  = merge_samples(rd_sample, phi_sample)
    return merge_samples(rd_sample, tht_sample)                                       # (resolution, sp_per_dim ^ 3, 4)

def phase_hg_nodiv(g: float, samples: torch.Tensor):
    """ Evaluate Henyey-Greenstein phase function 
        Note that this function is not for spherical sampling but 2D ellipse sampling
        therefore the value for PDF should be divided by 2pi
    """
    interm = 1 + g ** 2 - 2 * g * samples
    sqrt_interm = torch.sqrt(interm)
    return (1 - g ** 2) / 2 / (interm * sqrt_interm)

def eval_sec_scat_cos(d: float, T: float, cos_samples: torch.Tensor):
    """ Evaluate the second scattering (at elliptical vertex) cosine term """
    x_scatter =  0.5 * (T + d) * (T - d) / (T - cos_samples * d)
    return (x_scatter - d * cos_samples) / (x_scatter - T), T - x_scatter
    
def eval_integrand(samples: torch.Tensor, g: float, eval_inv_sqr = True):
    """ Evaluate the elliptical sampling integrand
        f_p1 * f_p2 * inverse sqr
        samples: (alpha, d/T, phi, theta) samples of shape (resolution, sp_per_dim^3, 4)
        eval_inv_sqr: whether to account for inverse square distance
        return:  
        - ray_dirs (resolution, sp_per_dim^3, 3), local frame (defined by alpha and the ellipse main axis)
        - evaluated throughput (resolution, sp_per_dim^3) evaluation result
        
        Now, all the training samples can be obtained
    """
    resolution, n_samples, _ = samples.shape
    sin_thetas = torch.sin(samples[..., -1:])          # (N, SP_N, 1)
    ray_dirs   = torch.cat([
        torch.cos(samples[..., -2:-1]) * sin_thetas, 
        torch.sin(samples[..., -2:-1]) * sin_thetas, torch.cos(samples[..., -1:])
    ], dim = -1)           # (N, SP_N, 3)
    
    ori_dirs   = torch.zeros((resolution, n_samples, 3), **TENSOR_PROP)      # (N, SP_N, 3)
    ori_dirs[..., 0] = torch.sin(samples[..., 0])       # sin(alphas)
    ori_dirs[..., 2] = torch.cos(samples[..., 0])       # cos(alphas)
    
    cos_t1 = (ray_dirs * ori_dirs).sum(dim = -1)          # used in H-G phase function (N, SP_N)
    cos_t2, illum_lens = eval_sec_scat_cos(1, 1 / samples[..., 1], cos_t1)      # the second cosine, and the inverse sqaure distance
    throughput: torch.Tensor = phase_hg_nodiv(g, cos_t1) * phase_hg_nodiv(g, cos_t2)      # (Resolution, sp_per_dim^3)
    if eval_inv_sqr:
        throughput /= illum_lens * illum_lens
    
    # thoughputs should be normalized, this is actually related to Monte Carlo estimation
    # See 2020 paper Robust Fitting of Parallax-Aware Mixtures for Path Guiding and 
    # https://computergraphics.stackexchange.com/questions/13969/measure-or-jacobian-conversion-for-direction-sampling
    throughput *= torch.abs(sin_thetas.squeeze(dim = -1))
    throughput /= throughput.mean(dim = -1, keepdim = True)
    
    return ray_dirs, throughput
    
def generate_training_samples(
    idx_r, g = 0.7, resolution = 256, sp_per_dim = 8,
    r_start = 0.2, r_end = 0.999, alpha_start = -torch.pi, alpha_end = torch.pi,
    ret_dim4 = False, use_inv_sqr = True
):
    """ Get the training samples of a certain row

    Args:
        idx_r (_type_): row index (indexing d/T ratio axis)
        g (float, optional): phase function g. Defaults to 0.7.
        resolution (int, optional): grid resolution. Defaults to 256.
        sp_per_dim (int, optional): sample per dimension. Defaults to 8.

    Returns: use for EM training
        ray_dir: shape of (N, SP_N, 3)
        throughput: shape of (N, SP_N)
    """
    div_r = (r_end - r_start) / resolution
    div_a = (alpha_end - alpha_start) / resolution
    alphas = torch.arange(resolution, **TENSOR_PROP) * div_a + alpha_start
    r_s   = r_start + div_r * idx_r
    
    dim4_samples = generate_row(alphas[..., None], div_a, r_s, r_s + div_r, resolution, sp_per_dim)
    ray_dirs, throughput = eval_integrand(dim4_samples, g, use_inv_sqr)
    if ret_dim4:
        return dim4_samples, throughput
    return ray_dirs, throughput
    
    
