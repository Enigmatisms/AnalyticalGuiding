import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

def distance_sampling(n_samples: int, sigma_t: float):
    return -np.log(1 - np.random.rand(n_samples)) / sigma_t

def direction_sampling(n_samples):
    angles = np.random.rand(n_samples) * 2 * np.pi - np.pi
    return np.stack([np.cos(angles), np.sin(angles)], axis = 1)

def sample_generation():
    start_p = np.float32([0, 0])
    x_e     = np.float32([1, 1])
    scatter_k = 1
    n_samples = 500000
    sigma_t = 2
    
    trial_num = 1000
    means = []
    stds  = []
    for k in range(trial_num):
        ray_dirs       = np.zeros((n_samples, 2), dtype = np.float32)
        ray_dirs[:, 0] = 1
        ray_pos        = np.tile(start_p[None, :], reps = (n_samples, 1))
        for i in range(scatter_k):
            dists = distance_sampling(n_samples, sigma_t)
            ray_pos += ray_dirs * dists[:, None]
            ray_dirs = direction_sampling(n_samples)

        inverse_sqr = 1 / (np.linalg.norm(ray_pos - x_e[None, :], axis = -1) ** 2)
        mean = inverse_sqr.mean()
        std  = inverse_sqr.std()
        print(f"{k}: mean = {mean:.4f}, std = {std:.4f}")
        means.append(mean)
        stds.append(std)
    np.save(f"./cached/means-{scatter_k}-{trial_num}.npy", np.float32(means))
    np.save(f"./cached/stds-{scatter_k}-{trial_num}.npy", np.float32(stds))

def analysis():
    means = []
    stds  = []
    for i in range(1, 7):
        means.append(np.load(f'./cached/means-{i}-1000.npy'))
        stds.append(np.load(f'./cached/stds-{i}-1000.npy'))
        
    sns.set_style('whitegrid')
    plt.figure(figsize = (6, 4))
    font_path = '../pbrt-v3/utils/font/libertine.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # Set the loaded font as the global font
    plt.rcParams['font.family'] = prop.get_name()
    plt.rc('font', size=14)       
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 
    plt.rc('legend', fontsize=13) 
    plt.rc('figure', titlesize=12)
    plt.tight_layout()
    plt.subplots_adjust(left = 0.12, bottom = 0.14, right = 0.97, top = 0.98)     # bt 0.13 for single line ticks, 0.17 for double line ticks
    
    plt.xlabel('Number of scattering vertices')
    plt.ylabel('mean value distribution')
    
    plt.violinplot(means)
    plt.xticks(np.arange(6) + 1, np.arange(6) + 1)
    plt.yscale('log')
    plt.grid(True, which = "both", ls = "--")
    plt.savefig('./mean-distribution.png', dpi = 300)
    plt.show()

if __name__ == "__main__":
    analysis()
    
