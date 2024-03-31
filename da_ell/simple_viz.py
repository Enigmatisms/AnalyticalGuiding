import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = sys.argv[1]
    data = np.load(path)
    T_max     = 10
    
    dT_start = 0.2
    dT_end   = 0.999
    dT_div = (dT_end - dT_start) / data.shape[0]
    T_div  = T_max / data.shape[0]
    
    while True:
        try:
            input_str = input("Row and column to visualize (e.g: 0 3): ")
        except KeyboardInterrupt:
            print("Ctrl + C. Exiting...")
            break
        input_str = input_str.strip()
        if not input_str: break
        nums_str  = input_str.split(' ')
        row, col  = int(nums_str[0]), int(nums_str[1])
        to_viz    = data[row, col]
        
        dT_min    = dT_div * row + dT_start
        dT_max    = dT_div + dT_min
        T_min     = T_div * col
        T_max     = T_div + T_min
        plt.figure(figsize = (10, 7))
        plt.subplots_adjust(top = 0.96, hspace = 0.25, bottom = 0.08)
        plt.subplot(2, 1, 1)
        plt.title(f'CDF for d/T$\in [{dT_min:.4f}, {dT_max:.4f}]$, T$\in[{T_min:.3f}, {T_max:.3f}]$')
        plt.plot(np.linspace(0, np.pi, to_viz.shape[0]), to_viz)
        plt.xticks(np.linspace(0, np.pi, 7), labels = [f"{0 + 30 * i}" for i in range(7)])
        plt.xlabel('Theta')
        plt.ylabel('CDF')
        plt.grid(axis = 'both')
        
        plt.subplot(2, 1, 2)
        plt.title(f'PDF for d/T$\in [{dT_min:.4f}, {dT_max:.4f}]$, T$\in[{T_min:.3f}, {T_max:.3f}]$')
        plt.plot(np.linspace(0, np.pi, to_viz.shape[0] - 1), to_viz[1:] - to_viz[:-1])
        plt.xticks(np.linspace(0, np.pi, 7), labels = [f"{0 + 30 * i}" for i in range(7)])
        plt.xlabel('Theta')
        plt.ylabel('PDF')
        plt.grid(axis = 'both')
        plt.show()