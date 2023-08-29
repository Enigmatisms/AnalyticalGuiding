import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    origin = np.float32([0.0013011316, 0.0051353928, 0.0013929765, 0.0034081743, 0.0029452041, 0.0367359391])
    darts = np.float32([0.0000319566, 0.0000340005, 0.0000291367, 0.0000249658, 0.0000246023, 0.0000210247])
    origin = -np.log(origin)
    darts = -np.log(darts)
    ts = np.linspace(0.1, 0.6, 6)
    
    origin_df = pd.DataFrame.from_dict({'t':ts, 'val':origin})
    darts_df = pd.DataFrame.from_dict({'t':ts, 'val':darts})
    
    sns.scatterplot(data = origin_df, x = 't', y = 'val', s = 15)
    sns.scatterplot(data = darts_df, x = 't', y = 'val', s = 15)
    
    sns.lineplot(data = origin_df, x = 't', y = 'val', label = 'variance for original method')
    sns.lineplot(data = darts_df, x = 't', y = 'val', label = 'variance for DARTS')
    # plt.plot(ts, origin, color = 'red', label = 'variance for original method')
    # plt.plot(ts, darts, color = 'blue', label = 'variance for DARTS')
    plt.legend()
    plt.grid(axis = 'both')
    plt.xlabel('scattering coefficient')
    plt.ylabel('variance (log(1 + x))')
    plt.show()