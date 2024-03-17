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
import time
import torch
from generate_samples import generate_training_samples

INV_2PI = 0.5 / torch.pi

class MeshedVMM:
    def __init__(self, 
        resolution = 256, 
        num_comps  = 4, 
        iter_num   = 4, 
        epochs     = 1,  
        dtype = torch.float32, device = 'cuda:0'):
        self.mean_dirs = torch.rand((resolution, resolution, num_comps, 3), dtype = dtype, device = device)
        self.rems      = torch.rand((resolution, resolution, num_comps, 2), dtype = dtype, device = device)     # kappa and pi
        self.resolution = resolution
        self.num_comps  = num_comps
        self.iter_num   = iter_num
        self.epochs = epochs

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
    
    def max_step(self, row_idx: int, dir_samples: torch.Tensor, weights: torch.Tensor, evals: torch.Tensor):
        """ Maximization step of the weighted EM algorithm

        Args:
            row_idx (int): index of the processing row
            dir_samples (torch.Tensor): direction samples, of shape (256, 512, 3)
            weights (torch.Tensor): weights samples, of shape (256, 512)
            evals (torch.Tensor): evaluated soft assignment proba from exp_step, shape (256, 512, 4), 4 is the number of component
        """
        # (256, 512, 1, 1) * (256, 512, 4, 1) * (256, 512, 1, 3) -> (256, 512, 4, 3) --sum--> (256, 4, 3)
        weighted_eval = weights[..., None] * evals      # (256, 512, 4)
        rk      = (weighted_eval[..., None] * dir_samples.unsqueeze(-2)).sum(dim = 1)     # (256, 4, 3)
        lengths = rk.norm(dim = -1, keepdim = True)                               # (256, 4, 1)
        merged_eval = weighted_eval.sum(dim = -2)
        bar_rk  = lengths / merged_eval.unsqueeze(-1)             # (256, 4, 1)
        # update mean direction
        self.mean_dirs[row_idx] = rk / lengths                                 # (256, 4, 3)
        # update kappa (disperse)
        self.rems[row_idx, ..., :1]  = bar_rk * (1 + 2 / (1 - bar_rk * bar_rk))                    # kappa is approximated, (256, 4, 1)
        # update pi
        self.rems[row_idx, ..., -1:] = merged_eval / merged_eval.sum(dim = -1, keepdim = True)        # (256, 4) / (256, 1) -> (256, 4)
        
    def train_row(self, row_idx: int, dir_sample: torch.Tensor, weights: torch.Tensor):
        for _ in range(self.iter_num):
            v_evals = self.exp_step(row_idx, dir_sample)
            self.max_step(row_idx, dir_sample, weights, v_evals)
            
    def train(self, g = 0.7):
        start_time = time.time()
        for epoch in range(self.epochs):
            for row in range(self.resolution):
                dir_samples, throughput = generate_training_samples(row, g, resolution = self.resolution)
                self.train_row(row, dir_samples, throughput)
            print(f"Finished weighted EM training epoch {epoch} / {self.epochs}")
        print(f"Training completed. Time consumption: {time.time() - start_time:.4f} s")

if __name__ == "__main__":
    print("Weight EM unit test")