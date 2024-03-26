""" Sample from the trained VMM and visualize the result
    Calculate K-L divergence
    @author: Qianyue He
    @date:   2024-3-24
"""
import os
import tqdm
import torch
import numpy as np
import configargparse
import matplotlib.pyplot as plt
from typing import Tuple
from rich.console import Console
from generate_samples import TENSOR_PROP
from inv_trans import inverse_transform_sampling
from scipy.spatial.transform import Rotation as Rot

CONSOLE = Console(width = 128)

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')

    parser.add_argument("--batch_size",  default = 16384, help = "Samples per batch",                       type = int)
    parser.add_argument("--iterations",  default = 1000,  help = "Number of iterations",                    type = int)
    parser.add_argument("--viz_phi_res", default = 512,   help = "Phi dim visualization grid num",          type = int)
    parser.add_argument("--viz_tht_res", default = 256,   help = "Theta dim visualization grid num",        type = int)
    
    parser.add_argument("--phi_start",   default = "-torch.pi", help = "Alpha angle start", type = str)
    parser.add_argument("--phi_end",     default = "torch.pi",  help = "Alpha angle end",   type = str)
    parser.add_argument("--tht_start",   default = "0",         help = "Alpha angle start", type = str)
    parser.add_argument("--tht_end",     default = "torch.pi",  help = "Alpha angle end",   type = str)

    parser.add_argument("--save_folder", default = './histogram/',    help = "Saved model path", type = str)
    parser.add_argument("--save_name",   default = 'vmm_sample.png',  help = "Saved model name",  type = str)
    parser.add_argument("--load_folder", default = './models/',       help = "Loaded model path",  type = str)
    parser.add_argument("--load_name",   default = 'vmm_trained.npz', help = "Loaded model name",  type = str)

    if delayed_parse:
        return parser
    return parser.parse_args()

def load_vmm_param(path: str) -> Tuple[np.ndarray, torch.Tensor]:
    data = np.load(path)
    mean_dirs: np.ndarray   = data["mean_dirs"]
    rems: torch.Tensor      = torch.from_numpy(data["rems"]).to(**TENSOR_PROP)
    # return shape (N_comp, 3) (N_comp, 2)
    return mean_dirs.squeeze(), rems.squeeze()

def np_rotation_between(fixed: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
        fixed: 3D vector of R^3
        target: (N, 3) mean direction
    """
    fixed   = fixed[None, :].repeat(target.shape[0], axis = 0)
    axis = np.cross(fixed, target)          # (N, 3)
    dot = np.sum(fixed * target, axis = -1) # (N)
    
    # Not in-line, cross product is valid
    axis /= np.linalg.norm(axis, axis = -1, keepdims = True)
    axis *= np.arccos(dot)[..., None]
    Rs_np = Rot.from_rotvec(axis).as_matrix()
    return torch.from_numpy(Rs_np).to(**TENSOR_PROP)

def sample_vmm(
    Rs: torch.Tensor, rems: torch.Tensor, 
    phi_res: float, theta_res: float,
    batch_samp_num: int = 16384):
    pdf_comps     = rems[:, 1]              # pi

    indices = inverse_transform_sampling(pdf_comps, batch_samp_num)
    R       = Rs[indices]                     # shape (batch_samp_num, 3, 3)
    kappas  = rems[indices, 0]                # shape (batch_samp_num)
    
    # sample vMF
    phi     = torch.rand(batch_samp_num, **TENSOR_PROP) * 2 * torch.pi - torch.pi
    v_vecs  = torch.stack([torch.cos(phi), torch.sin(phi)], dim = -1)       # shape (batch_samp_num, 2)
    
    xi_samp = torch.rand(batch_samp_num, **TENSOR_PROP)
    ws      = 1 + torch.log(xi_samp + (1 - xi_samp) * torch.exp(-2 * kappas)) / kappas
    
    weighted_v = torch.sqrt(1 - ws * ws)[..., None] * v_vecs                # shape (batch_samp_num, 2)
    vmm_dirs = torch.cat([weighted_v, ws.unsqueeze(dim = -1)], dim = -1)    # shape (batch_samp_num, 3)

    # rotate to correct coordinate frame 
    vmm_dirs = R @ vmm_dirs[..., None]                                      # shape (batch_samp_num, 3, 1)
    vmm_dirs = vmm_dirs.squeeze()                                           # shape (batch_samp_num, 3)
    
    thetas   = torch.acos(vmm_dirs[..., -1]).clamp(0, torch.pi - 1e-6)      # z is cos_theta
    phis     = torch.atan2(vmm_dirs[:, 1], vmm_dirs[:, 0]).clamp(-torch.pi, torch.pi - 1e-6)
    
    phi_idxs = torch.floor((phis + torch.pi) / phi_res).to(torch.int32)
    tht_idxs = torch.floor(thetas / theta_res).to(torch.int32)
    return phi_idxs, tht_idxs 
    
def visualization(opts):
    path = os.path.join(opts.load_folder, opts.load_name)
    mean_dirs, rems = load_vmm_param(path)
    Rs = np_rotation_between(np.float32([0, 0, 1]), mean_dirs)
    
    phi_res = (opts.phi_end - opts.phi_start) / opts.viz_phi_res
    tht_res = (opts.tht_end - opts.tht_start) / opts.viz_tht_res
    
    histogram = torch.zeros((opts.viz_tht_res, opts.viz_phi_res), dtype = torch.int32, device = 'cuda:0')
    for _ in tqdm.tqdm(range(opts.iterations)):
        phi_idxs, tht_idxs = sample_vmm(Rs, rems, phi_res, tht_res, opts.batch_size)
        histogram[tht_idxs, phi_idxs] += 1
        
    histogram = histogram.to(torch.float32)
    histogram /= histogram.max()
    
    plt.tight_layout()
    plt.figure(figsize = (11, 5))
    plt.imshow(histogram.cpu().numpy(), aspect='auto')
    plt.colorbar()
    
    plt.xlabel("Phi (related to the ellipsoidal main axis)")
    plt.ylabel("Theta (related to the ellipsoidal main axis)")
    plt.xticks(np.linspace(0, opts.viz_phi_res, 13), labels = [f"{-180 + 30 * i}" for i in range(13)])
    plt.yticks(np.linspace(0, opts.viz_tht_res, 7),  labels = [f"{0 + 30 * i}" for i in range(7)])
    plt.grid(axis = 'both', alpha = 0.3)
    
    main_folder = f"./{opts.save_folder}"
    if os.path.exists(main_folder) == False:
        os.makedirs(main_folder)
    plt.savefig(os.path.join(main_folder, opts.save_name), dpi = 400)

if __name__ == "__main__":
    opts = get_options()
    opts.phi_start = eval(opts.phi_start)
    opts.phi_end   = eval(opts.phi_end)
    opts.tht_start = eval(opts.tht_start)
    opts.tht_end   = eval(opts.tht_end)
    visualization(opts)
    