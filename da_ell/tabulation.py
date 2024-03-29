"""
    DA Elliptical sampling tabulation generation
    @author: Qianyue He
    @date:   2024.3.29
    
    128 * 128  * 256 (sampling dimension)
"""
import os
import time
import tqdm
import torch
import numpy as np
import configargparse
from rich.console import Console

CONSOLE = Console(width = 128)
DISABLE_COMPILE = True
TENSOR_PROP = {'device':'cuda:0', 'dtype': torch.float32}

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("-r", "--resolution",
                                          default = 128,  help = "Resolution of the tabulation",            type = int)
    parser.add_argument("--mc_iter",      default = 4,    help = "Iteration number (1000 is 1s, normally)", type = int)
    parser.add_argument("--avg_smp_num",  default = 8,    help = "Sample per grid",                         type = int)
    parser.add_argument("--mc_smp_num",   default = 128,  help = "Monte Carlo integration sample number",   type = int)
    parser.add_argument("--quantize_num", default = 256,  help = "Sampling dimension quantization",         type = int)

    parser.add_argument("--sigma_a",      default = 0.0005, help = "Absorption coefficient", type = float)
    parser.add_argument("--sigma_s",      default = 0.3,    help = "Scattering coefficient", type = float)
    parser.add_argument("--hg_g",         default = 0.0,    help = "H-G phase value",        type = float)
    parser.add_argument("--T_max",        default = 10,     help = "Maximum target time",    type = float)

    parser.add_argument("--table_folder", default = './tables/',       help = "Saved model path", type = str)
    parser.add_argument("--table_name",   default = 'table.npy',       help = "Saved model name",  type = str)
    
    parser.add_argument("--disable_compile", default = False, action = "store_true", help = "Whether to disable torch.compile")

    if delayed_parse:
        return parser
    return parser.parse_args()

def condition_checking(x: torch.Tensor, name = 'Tensor', operator = torch.isnan, condition: str = 'NaN'):
    is_true = operator(x)
    if is_true.any():
        print(f"{name} has {condition}, ratio: {is_true.sum() / is_true.numel() * 100:.5f}%")

def ellipse_polar_distance(rd_sample: torch.Tensor):
    """ (T^2 - d^2) / 2 (T - d \cos \theta)
        input shape: (128, 256, 8, 3)
        returned shape (128, 256, 8, 1)
    """
    result = 0.5 * (rd_sample[..., 1] ** 2 - rd_sample[..., 0] ** 2) /            \
        (rd_sample[..., 1] - rd_sample[..., 0] * torch.cos(rd_sample[..., 2]))
    return result.unsqueeze(-1)

def generate_row(T_grid, T_div, r_s, r_e, 
        resolution = 128, avg_smp_num = 8, mc_smp_num = 128, quantize_num = 256):
    """ Generate random samples for one row
        row: same d/T ratio, different T (linspace)
        
        T_grid: the linspace of T. shape (128, 1)
        T_div:  spacing between T
        r_s:    start value of ratio d/T (row start)
        r_e:    end value of ratio d/T (row end)
        avg_smp_num: query dimension blurring sampling num
        mc_smp_num:  sampling dimension samples for MC integration. To decrease gmem usage, decrease the input value
        quantize_num: theta quantization
        
        return: returned size: (1) d, T, theta (128, 256 * 8, 3), (2) mc samples (128, 256 * 8, 128)
    """
    # samples for (d/T and T)
    rd_sample  = torch.rand(resolution, avg_smp_num, 2, **TENSOR_PROP)              # (128, 8, 2)
    
    # stratified sampling i
    theta_res  = torch.pi / quantize_num
    tht_qunat  = torch.arange(quantize_num, **TENSOR_PROP) * theta_res
    tht_qunat += torch.rand(quantize_num, **TENSOR_PROP) * theta_res
    tht_qunat  = tht_qunat[None, :, None].expand(resolution, -1, avg_smp_num)       # (128, 256, 8)
   
    rd_sample[..., 0] = rd_sample[..., 0] * (r_e - r_s) + r_s         # uniform in row d/T ratio start and ratio end  -> d/T
    rd_sample[..., 1] = rd_sample[..., 1] * T_div + T_grid            # uniform in [0, T] -> T
    # generate phi_theta_samples
    rd_sample[..., 0] *= rd_sample[..., 1]                                  # d/T converted to d
    rd_sample = rd_sample.unsqueeze(1).expand(-1, quantize_num, -1, -1)     # (128, 256, 8, 2)
    rd_sample = torch.cat([rd_sample, tht_qunat[..., None]], dim = -1)      # (128, 256, 8, 3) -> (d, T, theta)
    
    # stratified sampling ii (distributed more uniformly)
    mc_res      = 1 / mc_smp_num
    mc_samples  = torch.arange(mc_smp_num, **TENSOR_PROP) * mc_res
    mc_samples += torch.rand(mc_smp_num, **TENSOR_PROP) * mc_res                     # Monte Carlo integration estimation (128)
    polar_dists = ellipse_polar_distance(rd_sample)
    
    mc_samples = mc_samples[None, None, None, :].expand(resolution, quantize_num, avg_smp_num, -1)      # (1, 1, 1, 128) -> (128, 256, 8, 128)

    mc_samples = mc_samples * polar_dists                       # mc_sample (128, 256, 8, 128)
    
    # returned size: (1) d, T, theta (128, 256 * 8, 3), (2) mc samples (128, 256 * 8, 128)
    return rd_sample.reshape(resolution, -1, 3), mc_samples.reshape(resolution, -1, mc_smp_num)

def mc_integration(infos: torch.Tensor, samples: torch.Tensor, sigma_t: float, sigma_a: float, D: float):
    """
    Args:
        infos (torch.Tensor): shape (128, 256 * 8, 3)
        samples (torch.Tensor): shape (128, 256 * 8, 128)
        sigma_s (float): _description_
        sigma_a (float): _description_
    """
    D4 = 4 * D
    # d projected onto the sampling direction line (length)
    proj_len   = infos[..., 0] * torch.cos(infos[..., 1])     # (128, 256 * 8)
    # emitter to sampling direction line distance
    line_dist2 = infos[..., 0] ** 2 - proj_len ** 2           # (128, 256 * 8)
    # distance (^2) to the target vertex (or emitter)
    d2 = (proj_len[..., None] - samples) ** 2 + line_dist2[..., None] #  (128, 256 * 8, 128)
    # residual time (T - t)
    res_time = infos[..., 1:2] - samples                      # should all be greater than 0, shape (128, 256 * 8, 1) - (128, 256 * 8, 128)

    # 1 / [4piD(S - t)]^(3/2)
    # condition_checking(res_time, 'res_time', lambda x: x <= 0, 'Non-positive')
    coeff = (torch.pi * D4) * res_time
    # condition_checking(coeff, 'Coeff before sqrt')
    coeff = coeff * torch.sqrt(coeff)
    # condition_checking(coeff, 'Coeff after sqrt')
    # Lis is the evaluated DA * transmittance 
    # this can be fused to get faster, either through triton or CUDA
    # returned shape (128, 256)     # MC evaluated
    return (1 / coeff * torch.exp(- d2 / (res_time ** 2) / D4 - sigma_a * res_time - sigma_t * samples)).mean()

def tabulate_row(row_id: int, sigma_a: float, sigma_s: float,g : float, T_max: float, 
                mc_iter: 4, resolution = 128, avg_smp_num = 8, mc_smp_num = 128, quantize_num = 256) -> torch.Tensor:
    r_div   = 0.999 / resolution
    r_start = r_div * row_id
    r_end   = r_start + r_div
    T_div   = T_max / resolution
    T_grid  = torch.arange(resolution, **TENSOR_PROP).unsqueeze(-1)
    D = 1 / (3 * (sigma_a + sigma_s * (1 - g)))
    
    result = torch.zeros(resolution, quantize_num, **TENSOR_PROP)
    for _ in range(mc_iter):
        infos, mc_samples = generate_row(T_grid, T_div, r_start, r_end, resolution, avg_smp_num, mc_smp_num, quantize_num)
        result += mc_integration(infos, mc_samples, sigma_s + sigma_a, sigma_a, D)
    condition_checking(result, "MC integral")
    return result * (1 / mc_iter)

def tabulation(opts):
    table = torch.zeros(opts.resolution, opts.resolution, opts.quantize_num, **TENSOR_PROP)
    start_time = time.time()
    for row_id in tqdm.tqdm(range(opts.resolution)):
        # returned shape: (128, 256)
        row_result = tabulate_row(row_id, opts.sigma_a, opts.sigma_s, opts.hg_g, opts.T_max, opts.mc_iter, 
                                opts.resolution, opts.avg_smp_num, opts.mc_smp_num, opts.quantize_num)
        condition_checking(row_result, "row_result")
        # convert to CDF
        row_result = torch.cumsum(row_result, dim = -1)         # accumulation
        row_result /= row_result[:, -1:]                        # normalization
        table[row_id] = row_result
    
    CONSOLE.log(f"Tabulation completed. Time consumption: {time.time() - start_time:.2f} s")
    CONSOLE.rule()
    
    CONSOLE.log("Training statistics:")
    CONSOLE.log(f"\tHenyey-Greenstein phase function: g = {opts.hg_g}")
    CONSOLE.log(f"\tQuantization resolution: {opts.quantize_num}")
    CONSOLE.log(f"\tNumber of MC samples: {opts.mc_smp_num} × {opts.mc_iter} = {opts.mc_iter * opts.mc_smp_num}")
    CONSOLE.log(f"\tNumber of blurring samples: {opts.avg_smp_num} × {opts.mc_iter} = {opts.mc_iter * opts.avg_smp_num}")
    CONSOLE.log(f"\tTabulation resolution: {opts.resolution} × {opts.resolution}")
    CONSOLE.log(f"\tScattering & absorption coefficients: {opts.sigma_s:.4f} | {opts.sigma_a:.4f}")
    CONSOLE.log(f"\td/T upper bound: {opts.T_max}")
    
    CONSOLE.rule()
    
    if not os.path.exists(opts.table_folder):
        os.makedirs(opts.table_folder)
    path = os.path.join(opts.table_folder, opts.table_name)
    CONSOLE.log(f"Table exported to '{path}'.")
    
    table_np = table.cpu().numpy()
    np.save(path, table_np)

if __name__ == "__main__":
    torch.random.manual_seed(3407)
    opts = get_options()
    ellipse_polar_distance = torch.compile(ellipse_polar_distance, disable = opts.disable_compile)
    mc_integration = torch.compile(mc_integration, disable = opts.disable_compile)
    tabulation(opts)