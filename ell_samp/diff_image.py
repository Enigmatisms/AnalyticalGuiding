import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    
    image1 = np.load(path1)
    image2 = np.load(path2)
    h, w   = image1.shape
    
    plt.tight_layout()
    plt.figure(figsize = (11, 5))
    plt.imshow(image1 - image2, cmap = 'inferno', aspect='auto')
    plt.colorbar()
    
    plt.xlabel("Phi (related to the ellipsoidal main axis)")
    plt.ylabel("Theta (related to the ellipsoidal main axis)")
    plt.xticks(np.linspace(0, w, 13), labels = [f"{-180 + 30 * i}" for i in range(13)])
    plt.yticks(np.linspace(0, h, 7),  labels = [f"{0 + 30 * i}" for i in range(7)])
    plt.grid(axis = 'both', alpha = 0.3)
    plt.savefig(f"diff_image.png", dpi = 400)