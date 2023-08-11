""" A Monte Carlo estimator for a simple integrand
    Integrate this function: x^2 in domain [0, 2]
    using global exponential sampling
"""

import numpy as np

def cutoff_sample(cut_off, sigma, n_samples):
    alpha = 1. - np.exp(-sigma * cut_off)
    dists = -np.log(1 - np.random.rand(n_samples) * alpha) / sigma
    pdfs = sigma / alpha * np.exp(-sigma * dists)
    mask = dists < cut_off
    return dists[mask], pdfs

if __name__ == "__main__":
    # use exponential distribution exp(1)
    N_SAMPLES = 10000000
    sigma = 0.1

    eval_func = lambda x: 1 / (1 + x) / (1 + x)
    max_range = 1

    estimator = lambda x: eval_func(x) / (sigma * np.exp(- sigma * x))
    samples = -np.log(1 - np.random.rand(N_SAMPLES)) / sigma
    samples_rand = np.random.rand(N_SAMPLES) * max_range
    samples_cut, cut_pdfs = cutoff_sample(max_range, sigma, N_SAMPLES)

    mc_no_discard = estimator(samples)
    mc_in_range = (samples >= 0) & (samples <= max_range)

    mc_discard = mc_no_discard[mc_in_range]
    mc_no_discard[~mc_in_range] = 0
    mc_uniform = eval_func(samples_rand) * max_range
    mc_cut = eval_func(samples_cut) / cut_pdfs
    print(f"Simple 'discarding histogram': {mc_discard.mean()}")
    print(f"Simple 'no-discard histogram': {mc_no_discard.mean()}")
    print(f"Simple 'uniform rand': {mc_uniform.mean()}")
    print(f"Simple 'cutoff exp': {mc_cut.mean()}")