""" 
    Weighted EM training procedures
    introduced in Ruppert L, Herholz S, Lensch H P A. Robust fitting of parallax-aware 
    mixtures for path guiding[J]. ACM Transactions on Graphics (TOG), 2020, 39(4): 147: 1-147: 15.
    @author: Qianyue He
    @date:   2024.3.17

    component number should be the multiple of 4/8, best of power of 4/8
    4 component vMF MM, 4 component mounts to (3 + 1 + 1) * 4 = 20 floats, 5MB
    memory consumption is really nothing, we should care about training speed
"""


import os
import tqdm
import time
import torch
import numpy as np
import configargparse
from datetime import datetime
from rich.console import Console
from generate_samples import generate_training_samples
from torch.utils.tensorboard import SummaryWriter

INV_2PI = 0.5 / torch.pi
CONSOLE = Console(width = 128)
ANALYZE_CONVERGENCE = True

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("-r", "--resolution",
                                         default = 256,  help = "Resolution of the tabulation", type = int)
    parser.add_argument("--it",          default = 1000, help = "iteration number (1000 is 1s, normally)", type = int)
    parser.add_argument("--sp_per_dim",  default = 8,    help = "Sample per dimension (8, or 16)",         type = int)
    parser.add_argument("--num_comps",   default = 4,    help = "Number of VMM components",                type = int)
    parser.add_argument("--epochs",      default = 2,    help = "Number of epochs",                        type = int)

    parser.add_argument("--alpha_start", default = "-torch.pi",           help = "Alpha angle start", type = str)
    parser.add_argument("--alpha_end",   default = "torch.pi",            help = "Alpha angle end",   type = str)
    parser.add_argument("--r_start",     default = "0.25",                help = "Ratio value start", type = str)
    parser.add_argument("--r_end",       default = "1.0",                 help = "Ratio value end",   type = str)
    parser.add_argument("--g",           default = 0.7,                   help = "H-G phase value",   type = float)
    parser.add_argument("--lr",          default = 0.5,                   help = "Learning rate",     type = float)
    
    parser.add_argument("--save_folder", default = './models/',           help = "Saved model path", type = str)
    parser.add_argument("--save_name",   default = 'vmm_trained.npz',     help = "Saved model name",  type = str)
    parser.add_argument("--summary_mode", 
                                         default = 'datetime', choices=['datetime', 'none'], help = "Summary output mode",  type = str)

    parser.add_argument("--use_inv_sqr", default = False, action = "store_true", help = "Whether to use inverse square")
    parser.add_argument("--use_sec_phs", default = False, action = "store_true", help = "Whether to use the second phase function")

    if delayed_parse:
        return parser
    return parser.parse_args()

def numerical_guard(tensor, threshold=1e-5):
    return torch.where(torch.abs(tensor) < threshold, torch.sign(tensor) * threshold, tensor)

class MeshedVMM:
    def __init__(self, 
        resolution = 256, 
        num_comps  = 4, 
        iter_num   = 4, 
        epochs     = 1,  
        log_mode   = "./logs/",
        lr         = 0.5,
        dtype = torch.float32, device = 'cuda:0'):
        
        samples = torch.rand(resolution, resolution, num_comps, 2, dtype = dtype, device = device)
        sin_thetas = torch.sin(samples[..., -1:])
        # we can have better initialization
        self.mean_dirs   = torch.cat([
            torch.cos(samples[..., -2:-1]) * sin_thetas, 
            torch.sin(samples[..., -2:-1]) * sin_thetas, torch.cos(samples[..., -1:])
        ], dim = -1)           # make sure the initialized directions are valid
        
        self.rems       = torch.rand(resolution, resolution, num_comps, 2, dtype = dtype, device = device) * 0.5    # kappa and pi
        self.rems[..., -1] += 0.5
        self.rems[..., -1:] /= self.rems[..., -1:].sum(dim = -2, keepdim = True)     # (256, 4, 1) / (256, 1, 1) normalization
        
        self.resolution = resolution
        self.num_comps  = num_comps
        self.iter_num   = iter_num
        self.epochs     = epochs
        self.lr         = lr
        self.dtype      = dtype
        self.device     = device
        if ANALYZE_CONVERGENCE:
            summary_path = 'logs'
            if log_mode == 'datetime':
                summary_path = os.path.join(summary_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.writer = SummaryWriter(summary_path)

    def exp_step(self, row_idx: int, samples: torch.Tensor) -> torch.Tensor:
        """ The input will be (N, SP_N, 3), N is the number of VMM to be trained in a row, SP_N is the sample count 
            e.g: input shape (256, 512, 3) (direction samples), we have 4 component each grid
            each component maps (512, 3) to (512, 1), therefore the output is (256, 512, 4)
            to best make use of tensor parallelism, easy enough
            
            This is the soft assignment step
            
            Return:
                tensor of shape (256, 512, 4): 256 grid, 512 samples, evaluated by 4 different components
        """
        mu    = self.mean_dirs[row_idx].unsqueeze(1)     # (256, 1,   4, 3)
        kappa = self.mean_dirs[row_idx, :, :, 0].unsqueeze(-2)  # (256, 1, 4)
        pis   = self.mean_dirs[row_idx, :, :, 1].unsqueeze(-2)  # (256, 1, 4)
        samples = samples.unsqueeze(-2)                  # (256, 512, 1, 3)
        # coeff = INV_2PI * kappa / (1 - torch.exp(-2 * kappa))       # (256, 1, 4)
        # mu_t_w  = (mu * samples).sum(dim = -1)         # (256, 512, 4)
        # the following: pi_k * vMF_pdf(w)
        v_evals = INV_2PI * kappa * torch.exp(kappa * ((mu * samples).sum(dim = -1) - 1)) / (1 - torch.exp(-2 * kappa)) * pis
        # normalization, the last dim then will sum to 1
        return v_evals / v_evals.sum(dim = -1, keepdim = True)
    
    def max_step(self, row_idx: int, dir_samples: torch.Tensor, weights: torch.Tensor, evals: torch.Tensor, it_idx: int):
        """ Maximization step of the weighted EM algorithm

        Args:
            row_idx (int): index of the processing row
            dir_samples (torch.Tensor): direction samples, of shape (256, 512, 3)
            weights (torch.Tensor): weights samples, of shape (256, 512)
            evals (torch.Tensor): evaluated soft assignment proba from exp_step, shape (256, 512, 4), 4 is the number of component
        """
        # (256, 512, 1, 1) * (256, 512, 4, 1) * (256, 512, 1, 3) -> (256, 512, 4, 3) --sum--> (256, 4, 3)
        weighted_eval = weights[..., None]  * evals      # (256, 512, 4)
        rk      = (weighted_eval[..., None] * dir_samples.unsqueeze(-2)).sum(dim = 1)     # (256, 4, 3)
        lengths = rk.norm(dim = -1, keepdim = True)                                # (256, 4, 1)
        merged_eval = weighted_eval.sum(dim = -2)                                  # (256, 4)
        bar_rk  = (lengths / merged_eval[..., None]).squeeze(dim = -1)             # (256, 4)
        if ANALYZE_CONVERGENCE:
            # calculate the change of variables
            mean_dirs       = rk / lengths
            old_dirs        = self.mean_dirs[row_idx]
            dirs_param_diff = torch.abs(mean_dirs - old_dirs).mean()     
            self.mean_dirs[row_idx]  = mean_dirs * self.lr + old_dirs * (1 - self.lr)
            self.mean_dirs[row_idx] /= self.mean_dirs[row_idx].norm(dim = -1, keepdim = True)
            
            new_remes  = torch.stack([
                bar_rk * (1 + 2 / numerical_guard(1 - bar_rk * bar_rk)),
                merged_eval / merged_eval.sum(dim = -1, keepdim = True)
            ], dim = -1)
            rems_param_diff    = torch.abs(new_remes - self.rems[row_idx]).mean(dim = (0, 1))       # (resolution, num_comps, 2) -> 2
            self.rems[row_idx] = new_remes * self.lr + self.rems[row_idx] * (1 - self.lr)
            
            time_stamp = it_idx + row_idx * self.iter_num
            self.writer.add_scalar('Param-Diff/Direction', dirs_param_diff,    time_stamp)
            self.writer.add_scalar('Param-Diff/Kappa',     rems_param_diff[0], time_stamp)
            self.writer.add_scalar('Param-Diff/Pi',        rems_param_diff[1], time_stamp)
        else:
            self.mean_dirs[row_idx] = (rk / lengths) * self.lr + self.mean_dirs[row_idx] * (1 - self.lr)
            self.mean_dirs[row_idx] /= self.mean_dirs[row_idx].norm(dim = -1, keepdim = True)
            new_remes  = torch.stack([
                bar_rk * (1 + 2 / numerical_guard(1 - bar_rk * bar_rk)),            # update kappa (disperse), approximated, (256, 4)
                merged_eval / merged_eval.sum(dim = -1, keepdim = True)             # update pi (256, 4) / (256, 1) -> (256, 4)
            ], dim = -1)
            self.rems[row_idx] = new_remes * self.lr + self.rems[row_idx] * (1 - self.lr)
        
    def train_row(self, row_idx: int, dir_sample: torch.Tensor, weights: torch.Tensor):
        for it_idx in range(self.iter_num):
            v_evals = self.exp_step(row_idx, dir_sample)
            self.max_step(row_idx, dir_sample, weights, v_evals, it_idx = it_idx)
            
    def save_params(self, path = 'model.npz'):
        mean_dirs = self.mean_dirs.cpu().numpy()
        rems      = self.rems.cpu().numpy()
        np.savez(path, mean_dirs = mean_dirs, rems = rems)
        CONSOLE.log(f"Model saved to '{path}'")
        
    def load_params(self, path = 'model.npz'):
        data = np.load(path)
        self.mean_dirs = torch.from_numpy(data['mean_dirs']).to(device = self.device, dtype = self.dtype)
        self.rems      = torch.from_numpy(data['rems']).to(device = self.device, dtype = self.dtype)
        CONSOLE.log(f"Model loaded from '{path}'")
            
    def train(self, opts):
        opts.alpha_start = eval(opts.alpha_start)
        opts.alpha_end   = eval(opts.alpha_end)
        opts.r_start     = eval(opts.r_start)
        opts.r_end       = eval(opts.r_end)

        start_time = time.time()
        for epoch in range(self.epochs):
            for row in tqdm.tqdm(range(self.resolution)):
                dir_samples, throughput = generate_training_samples(row, opts.g, self.resolution, opts.sp_per_dim, opts.r_start, 
                                            opts.r_end, opts.alpha_start, opts.alpha_end, False, opts.use_inv_sqr, opts.use_sec_phs)
                self.train_row(row, dir_samples, throughput)
            CONSOLE.log(f"Finished weighted EM training epoch {epoch} / {self.epochs}")
        CONSOLE.log(f"Training completed. Time consumption: {time.time() - start_time:.4f} s")
        CONSOLE.rule()
        CONSOLE.log("Training statistics:")
        CONSOLE.log(f"\tHenyey-Greenstein phase function: g = {opts.g}")
        CONSOLE.log(f"\tNumber of samples per VMM: {opts.sp_per_dim ** 3}")
        CONSOLE.log(f"\tTabulation resolution: {self.resolution} × {self.resolution}")
        CONSOLE.log(f"\tNumber of iterations: {opts.it}")
        CONSOLE.log(f"\tNumber of epochs: {opts.epochs}")
        CONSOLE.log(f"\tConvergence statistics: TODO")
        
if __name__ == "__main__":
    torch.random.manual_seed(3407)
    opts = get_options()
    
    vmm_grid = MeshedVMM(
        resolution = opts.resolution,
        num_comps  = opts.num_comps,
        iter_num   = opts.it,
        epochs     = opts.epochs,
        log_mode   = opts.summary_mode,
        lr         = opts.lr
    )

    vmm_grid.train(opts)
    
    if not os.path.exists(opts.save_folder):
        os.makedirs(opts.save_folder)
    path = os.path.join(opts.save_folder, opts.save_name)
    vmm_grid.save_params(path)
    vmm_grid.writer.close()
    