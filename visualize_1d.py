import numpy as np
import matplotlib.pyplot as plt
from options import get_options
from mh import MetropolisHasting
from da import *
from colors import COLORS
from functools import partial

"""
TODO:
(1) Try non-exact temporal sampling case
(2) Implement all other simpler DEs, and see if our result has consistency
(3) Implement 2D visualization (Taichi, 2D) and 2D sampling path 
    - path visualization is suitable for direction sampling, for distance sampling
    - maybe some better visualization method should be used
"""

solutions = {"h_n": half_diffusion, "h_d": partial(half_diffusion, sub = True), "full": full_diffusion}

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

def forward_eval_func(opts, func, t_plus: float, D: float, c: float):
    return lambda x: tr(x - opts.xmin, opts.ua, opts.us) * \
        func(x, t_plus + opts.eps - x, 0, opts.eps, opts.ua, D, c)

def backward_eval_func(opts, func, t_plus: float, D: float, c: float):
    return lambda x: tr(x - opts.xmin, opts.ua, opts.us) * \
        func(x, t_plus + opts.xmin - opts.eps - (x - opts.xmin), 0, opts.eps, opts.ua, D, c)

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
        travel_dist = xs - opts.xmin
        trs = tr(travel_dist, opts.ua, opts.us)
        D = get_diffusion_length(opts.ua, opts.us)
        print(f"Diffusion length: {D}, time points to compare: {opts.t_plus_num}, time added: {opts.t_plus_val}")

        diffusion_solution = solutions[opts.func]
        solutions       = []
        results         = []
        solution_labels = []
        result_labels   = []
        # For the convenience of making better comparison
        for i in range(opts.t_plus_num + 1):
            t_plus = i * opts.t_plus_val
            ts_use = ts + t_plus 
            solution = diffusion_solution(xs, ts_use, 0, opts.eps, opts.ua, D, c)
            result = trs * solution
            result /= result.sum()
            solutions.append(solution)
            results.append(result)
            solution_labels.append(f"DA, tplus = {t_plus:.3f}")
            result_labels.append(f"Tr * DA, tplus = {t_plus:.3f}")

        # Normalize by summation (approximate integral normalization)
        normalized_trs = trs / trs.sum()
        plt.figure(0)
        plt.plot(xs, normalized_trs, color = COLORS[0], label = 'Tr')
        for i, (result, label) in enumerate(zip(results, result_labels)):
            plt.plot(xs, result, c = COLORS[i + 1], label = label)
        plt.legend()
        plt.title("RTS and traditional sampling curve comparison")
        plt.grid(axis = 'both')

        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(xs, trs, c = 'r', label = 'tr')
        plt.title("Transmittance")
        plt.grid(axis = 'both')

        plt.subplot(2, 1, 2)
        for i, (solution, label) in enumerate(zip(solutions, solution_labels)):
            plt.plot(xs, solution /solution.sum(), c = COLORS[i], label = label)
        plt.title("DA solutions: To be investigated (more)")
        plt.legend()
        plt.grid(axis = 'both')

        plt.figure(2)
        x_ticks = [1, 2]
        labels = ['Transmittance', 'Metropolis']
        if opts.backward:
            eval_func = backward_eval_func(opts, diffusion_solution, 0, D, c)
        else:
            eval_func = forward_eval_func(opts, diffusion_solution, 0, D, c)
        mh_samples = MetropolisHasting.get_samples(eval_func, opts.ua + opts.us, opts.snum, opts.xmin, opts.xmax)
        old_samples = transmittance_sampling(opts.snum, opts.ua, opts.us) + opts.xmin

        collections = [old_samples, mh_samples]
        for i, result in enumerate(results):
            integral = normalized_integral(result)
            new_samples = inverse_sampling(integral, xs, opts.snum)
            x_ticks.append(i + 3)
            collections.append(new_samples)
            labels.append(f"RTS, t={i * opts.t_plus_val:.3f}")

        plt.title("Samples visualization")
        plt.violinplot(collections, showmeans = True)
        plt.xticks(x_ticks, labels)
        plt.grid(axis = 'both')
        plt.show()
    else:                       # fix a temporal point and examine the intensity changes induced by time changes
        raise NotImplementedError("This branch is not yet implemented")