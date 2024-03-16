import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def erf(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = np.sign(x)
    x = np.abs(x)

    t = 1 / (1 + p * x)
    t2 = t * t
    t4 = t2 * t2
    y = 1 - (a5 * t * t4 + a4 * t4 + a3 * t2 * t + a2 * t2 + a1 * t) * np.exp(-x * x)

    return sign * y

def erf_gaussian(x, mu = 0, std = 1):
    return 0.5 * (1 + erf((x - mu) / std / np.sqrt(2)))

def exact_gaussian(x, mu = 0, std = 1):
    return norm.cdf(x, loc = mu, scale = std)

import numpy as np
import matplotlib.pyplot as plt

def box_muller_1d(n):
    # Generate n pairs of independent uniform random variables
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)

    # Box-Muller transform for 1D Gaussian
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    return z

def show_box_muller():
    # Generate 10,000 random numbers using Box-Muller for 1D Gaussian
    num_samples = 10000
    data = box_muller_1d(num_samples)

    # Plot a histogram to visualize the distribution
    plt.hist(data, bins=50, density=True, alpha=0.75, color='blue', label='Generated Data')
    plt.title('Box-Muller Transform: 1D Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()
    
def test_approx_erf():
    mu  = 0
    std = 1 
    
    points = np.linspace(0, 3, 200)
    
    erf_gauss   = erf_gaussian(points, mu, std)
    exact_gauss = exact_gaussian(points, mu, std)
    
    plt.plot(points, erf_gauss, label = 'ERF')
    plt.plot(points, exact_gauss, label = 'Exact')
    plt.legend()
    plt.grid(axis = 'both')
    
    plt.show()

if __name__ == "__main__":
    show_box_muller()
    
    

    