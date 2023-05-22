import numpy as np
import matplotlib.pyplot as plt
from options import get_options
from mh import MetropolisHasting
from da import *

"""
TODO:
(1) Try non-exact temporal sampling case
(2) Implement all other simpler DEs, and see if our result has consistency
(3) Implement 2D visualization (Taichi, 2D) and 2D sampling path 
    - path visualization is suitable for direction sampling, for distance sampling
    - maybe some better visualization method should be used
"""

def tr(x, ua, us):
    return np.exp(- (ua + us) * x)

def get_diffusion_length(u_a, u_s, g = 0):
    return 1 / (3 * (u_a + (1 - g) * u_s))

def normalized_integral(values: np.ndarray) -> np.ndarray:
    results = np.cumsum(values)
    results /= results.max()
    return results

def inverse_sampling(table: np.ndarray, pos_lut: np.ndarray, sample_num: int) -> np.ndarray:
    assert table.shape[0] == pos_lut.shape[0]
    rands = np.random.uniform(0, 1, sample_num)
    table_max = table.shape[0]
    indices = np.searchsorted(table, rands, side = "right").clip(0, table_max)
    return pos_lut[indices]

def transmittance_sampling(sample_num:int, ua: float, us: float):
    ut = ua + us
    xs = np.random.uniform(0, 1, sample_num)
    return -np.log(1 - xs) / ut

def forward_eval_func(opts, D: float, c: float):
    return lambda x: tr(x - opts.xmin, opts.ua, opts.us) * \
        half_diffusion_neumann(x, opts.eps - x, 0, opts.eps, opts.ua, D, c)

def backward_eval_func(opts, D: float, c: float):
    return lambda x: tr(x - opts.xmin, opts.ua, opts.us) * \
        half_diffusion_neumann(x, opts.xmin - opts.eps - (x - opts.xmin), 0, opts.eps, opts.ua, D, c)

if __name__ == "__main__":
    opts = get_options()
    c = SOL if opts.sol == "physical" else 1
    if opts.mode == "time":     # fix a spatial point and examine the intensity changes induced by time changes 
        raise NotImplementedError("This branch is not yet implemented")
    elif opts.mode == 'rts':
        # descending time
        xs = np.linspace(opts.xmin, opts.xmax, opts.pnum)
        if opts.backward:
            # Backward setup: emitter is closer to 0 (for e.g, 0.05)
            # The sampling direction is along the positive x-axis
            # According to my ideas:
            ts = opts.xmin - opts.eps - (xs - opts.xmin)
        else:
            ts = np.linspace(opts.eps - opts.xmin, opts.eps - opts.xmax, opts.pnum)
        ts += opts.t_plus
        travel_dist = xs - opts.xmin
        trs = tr(travel_dist, opts.ua, opts.us)
        D = get_diffusion_length(opts.ua, opts.us)
        solution = half_diffusion_neumann(xs, ts, 0, opts.eps, opts.ua, D, c)
        result = trs * solution
        print(f"Diffusion length: {D}, time added: {opts.t_plus}")

        # Normalize by summation (approximate integral normalization)
        result /= result.sum()
        normalized_trs = trs / trs.sum()
        plt.figure(0)
        plt.plot(xs, result, c = 'r', label = 'tr * diffusion solution')
        plt.plot(xs, normalized_trs, c = 'b', label = 'tr')
        plt.legend()
        plt.title("RTS and traditional sampling curve comparison")
        plt.grid(axis = 'both')

        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(xs, trs, c = 'r', label = 'tr')
        plt.title("Transmittance")
        plt.grid(axis = 'both')

        plt.subplot(2, 1, 2)
        plt.plot(xs, solution, c = 'r', label = 'solution')
        plt.title("DA solution")
        plt.legend()
        plt.grid(axis = 'both')

        plt.figure(2)
        integral = normalized_integral(result)
        trs_integral = normalized_integral(trs)
        new_samples = inverse_sampling(integral, xs, opts.snum)
        if opts.backward:
            eval_func = backward_eval_func(opts, D, c)
        else:
            eval_func = forward_eval_func(opts, D, c)
        mh_samples = MetropolisHasting.get_samples(eval_func, opts.ua + opts.us, opts.snum, opts.xmin, opts.xmax)
        # old_samples = inverse_sampling(trs_integral, xs, opts.snum)
        old_samples = transmittance_sampling(opts.snum, opts.ua, opts.us) + opts.xmin

        collections = [new_samples, old_samples, mh_samples]
        plt.title("Samples visualization")
        plt.violinplot(collections)
        plt.xticks([1, 2, 3], ['RTS samples', 'Transmittance samples', 'Metropolis Hasting'])
        plt.grid(axis = 'both')
        plt.show()
    else:                       # fix a temporal point and examine the intensity changes induced by time changes
        raise NotImplementedError("This branch is not yet implemented")