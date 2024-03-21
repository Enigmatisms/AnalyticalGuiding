""" 
    Visualize the samples in a grid (form a 2D image)
    @author: Qianyue He
    @date:   2024.3.17
    
    TODO
"""

import os
import tqdm
import time
import torch
import numpy as np
import configargparse
import matplotlib.pyplot as plt
from generate_samples import generate_training_samples, TENSOR_PROP, PI_2
from rich.console import Console

CONSOLE = Console(width = 128)

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("-r", "--resolution",
                                     default = 1, help = "Resolution of the tabulation (for visualization, set 1)", type = float)

    parser.add_argument("--it",          default = 1000, help = "iteration number (1000 is 1s, normally)", type = int)
    parser.add_argument("--sp_per_dim",  default = 8,    help = "Sample per dimension (8, or 16)",         type = int)
    parser.add_argument("--viz_phi_res", default = 512,  help = "Phi dim visualization grid num",          type = int)
    parser.add_argument("--viz_tht_res", default = 256,  help = "Theta dim visualization grid num",        type = int)

    parser.add_argument("--alpha_start", default = "torch.pi / 3",        help = "Alpha angle start", type = str)
    parser.add_argument("--alpha_end",   default = "torch.pi / 3 + 1e-2", help = "Alpha angle end",   type = str)
    parser.add_argument("--r_start",     default = "0.5",                 help = "Ratio value start", type = str)
    parser.add_argument("--r_end",       default = "0.5 + 1e-3",          help = "Ratio value end",   type = str)
    parser.add_argument("--g",           default = 0.7,                   help = "H-G phase value",   type = float)
    
    parser.add_argument("--npy_name",    default = 'cached.npy',          help = "Cached numpy filename", type = str)
    parser.add_argument("--png_name",    default = 'mc_results.png',      help = "Saved image filename",  type = str)

    parser.add_argument("--use_inv_sqr", default = False, action = "store_true", help = "Whether to use inverse square")

    if delayed_parse:
        return parser
    return parser.parse_args()

def visualization(opts):
    opts.alpha_start = eval(opts.alpha_start)
    opts.alpha_end   = eval(opts.alpha_end)
    opts.r_start     = eval(opts.r_start)
    opts.r_end       = eval(opts.r_end)
    result = torch.zeros((opts.viz_tht_res, opts.viz_phi_res), **TENSOR_PROP)
    sum_time = 0
    for _ in tqdm.tqdm(range(opts.it)):
        start_time = time.time()
        dim4_samples, throughput = generate_training_samples(0, opts.g, 1, 8, opts.r_start, opts.r_end, opts.alpha_start, opts.alpha_end, True, opts.use_inv_sqr)
        sum_time  += time.time() - start_time
        
        # returned dim4_samples is of shape: (1, 512, 4)
        phi_thetas = dim4_samples[0, :, -2:]        # (512, 2)
        throughput = throughput.squeeze()
        phi_thetas[:, 0] = ((phi_thetas[:, 0] + torch.pi) / PI_2 * opts.viz_phi_res)
        phi_thetas[:, 1] = (phi_thetas[:, 1] / torch.pi * opts.viz_tht_res)
        phi_thetas = phi_thetas.to(torch.int32)
        result[phi_thetas[:, 1], phi_thetas[:, 0]] += throughput
    sum_time /= opts.it
    CONSOLE.log(f"Generate training samples ({opts.sp_per_dim ** 3} samples, {opts.it} iterations): {sum_time * 1000} ms avg.")
    result   /= opts.it
    np_result = result.cpu().numpy()

    deg_alpha = int(opts.alpha_start * 180 / torch.pi)
    deg_str   = f"{deg_alpha}" if deg_alpha >= 0 else f"n{deg_alpha}"
    main_folder = f"./dT-{opts.r_start}-{deg_str}"
    if os.path.exists(f"{main_folder}/caches") == False:
        os.makedirs(f"{main_folder}/caches")
    
    g_str = f"{opts.g}" if opts.g > 0 else f"n{-opts.g}"
    np.save(f"{main_folder}/caches/cached-{g_str}.npy", np_result)

    plt.tight_layout()
    plt.figure(figsize = (11, 5))
    plt.imshow(np_result, aspect='auto')
    plt.colorbar()
    
    plt.xlabel("Phi (related to the ellipsoidal main axis)")
    plt.ylabel("Theta (related to the ellipsoidal main axis)")
    plt.xticks(np.linspace(0, opts.viz_phi_res, 13), labels = [f"{-180 + 30 * i}" for i in range(13)])
    plt.yticks(np.linspace(0, opts.viz_tht_res, 7),  labels = [f"{0 + 30 * i}" for i in range(7)])
    plt.grid(axis = 'both', alpha = 0.3)
    plt.savefig(f"{main_folder}/mc_results-{g_str}.png", dpi = 400)
    
if __name__ == "__main__":
    torch.random.manual_seed(3407)
    opts = get_options()
    visualization(opts)
    