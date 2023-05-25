""" Metropolis Hasting algorithm
"""

import random
import numpy as np
import matplotlib.pyplot as plt

class MetropolisHasting:
    """ We will do a distance sampling (one side unbounded)
        and we will need a local exploration mutation strategy
        and a global restart mutation
        Global restart mutation can not be symmetric but local can
    """
    def __init__(self, func, 
        ut: float, 
        global_ratio: float = 0.1, 
        min_range: float = 0.0,
        max_range: float = 10
    ):
        self.ut = ut
        self.mfp = 1 / ut
        self.min_range = min_range
        self.max_range = max_range

        self.state, pdf = self.global_mutate()
        self.w = self.state / pdf
        self.old_value = func(self.state)
        self.global_ratio = global_ratio

    def global_mutate(self):
        return random.uniform(self.min_range, self.max_range), 1 / (self.max_range - self.min_range)
    
    def local_mutate(self):
        """ Return uniformly jittered sample and pdf (this is non-symmetric) """
        half_mfp = self.mfp * 0.5
        lb = max(self.state - half_mfp, self.min_range)
        ub = self.state + half_mfp
        return random.uniform(lb, ub), 1 / (ub - lb)
    
    def sample(self, func):
        if random.uniform(0, 1) < self.global_ratio:
            sample, _ = self.global_mutate()
            tr_ratio = 1
        else:
            sample, tr_to_new = self.local_mutate()
            half_mfp = self.mfp * 0.5
            lb = max(sample - half_mfp, self.min_range)
            ub = sample + half_mfp
            tr_from_new = 1 / (ub- lb)      # from new sample to old state, sample pdf
            tr_ratio = tr_from_new / tr_to_new
        cur_value = func(sample)
        thresh = min(1, cur_value / self.old_value * tr_ratio)
        rand_val = random.uniform(0, 1)
        if rand_val < thresh:
            self.state = sample
            self.old_value = cur_value
        return self.state
    
    @staticmethod
    def get_samples(func, ut: float, sample_num: int, min_range: float = 0., max_range: float = 10.):
        mh_sampler = MetropolisHasting(func, ut, 0.1, min_range, max_range)
        samples = []
        for _ in range(sample_num):
            sample = mh_sampler.sample(func)
            samples.append(sample)
        return np.float32(samples)
    
if __name__ == "__main__":
    # simple test

    func = lambda x: np.exp(- np.sqrt(2 * x + 1 / (x + 1)))
    mh_sampler = MetropolisHasting(func, 1, 0.1)
    samples = []
    for i in range(10000):
        sample = mh_sampler.sample(func)
        samples.append(sample)
    bin_cnts, _, _ = plt.hist(samples, np.linspace(0, 10, 100), label = 'Metropolis samples')
    xs = np.linspace(0, 10, 1000)
    ys = func(xs)
    ys = ys / (ys.max()) * bin_cnts.max()

    plt.plot(xs, ys, label = "Function to sample from")
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()
