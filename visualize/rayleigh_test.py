import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.legend_handler import HandlerPatch, HandlerBase
from matplotlib.transforms import Affine2D

# Define parameters for the Rayleigh functions
x = np.linspace(0, 5, 1000)
scale_factors = [1, 2, 3]  # Different peak heights


class HandlerRectangle(HandlerPatch):
    def __init__(self, color, fill_alpha):
        super().__init__()
        self.color = color
        self.alpha = fill_alpha
        
    def create_artists(self, _1, orig_handle, _2, _3, _4, height, _5, trans):
        w = 27                          # Width of the rectangle
        h = height
        line = Rectangle(xy=(0, h), width=w, height=0.1, linewidth = 1.0, edgecolor = self.color, label=orig_handle.get_label())
        rect = Rectangle(xy=(0, 0), width=w, height=h, facecolor = self.color, 
                edgecolor = (0, 0, 0, 0), linewidth=0, label=orig_handle.get_label(), alpha = self.alpha)
        line.set_transform(trans)
        rect.set_transform(trans)
        return [line, rect]
    
class HandlerVerticalLine(HandlerBase):
    def __init__(self, color):
        super().__init__()
        self.color = color
        
    def create_artists(self, legend, orig_handle, _1, _2, width, height, _3, trans):
        x = 0.5 * width
        line = plt.Line2D([x, x], [-0.25 * height, 1.5 * height], color = self.color, linestyle = '--')
        self.update_prop(line, orig_handle, legend)
        line.set_transform(trans)
        return [line]

def get_rayleigh(x: np.ndarray, scale: float) -> np.ndarray:
    return x / scale**2 * np.exp(-x**2 / (2 * scale**2))

def draw_sample(ax, shift, color):
    vertices = np.array([[-0.04, 0], [0, 0.2], [0.04, 0]])
    vertices[:, 0] += shift
    face = Polygon(vertices, closed=True, facecolor = color, alpha = 0.9)
    edge = Polygon(vertices, closed=True, edgecolor = 'grey', fill = False, alpha = 0.9)
    ax.add_patch(face)
    ax.add_patch(edge)

colors = ('#3559E0', '#BF3131', '#508D69', '#E48F45')

if __name__ == "__main__":
    y_scale       = 1.25
    up_limits     = [8, 7, 5] 
    shifts        = [1, 2, 4]
    scale_factors = [2, 1.5, 1]  # Different peak heights
    sample_position = [0.7, 2.2, 5.2]

    sns.set_style('whitegrid')
    
    font_path = './font/libertine.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rc('font', size=14)       
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    plt.rc('legend', fontsize=15) 
    plt.rc('figure', titlesize=12)
    
    fig, ax = plt.subplots(figsize = (8, 5.5))
    fig.subplots_adjust(left = 0.04, right = 0.96, top = 0.96, bottom = 0.04)
    
    for i in range(1, 6):
        ax.axhline(y = 0.2 * i, xmax = 0.98, linewidth=0.5, color='gray', alpha=0.5)

    plots = []
    handler_map = {}
    for i, (up_lim, scale, shift) in enumerate(zip(up_limits, scale_factors, shifts)):
        x = np.linspace(0, up_lim, 2000)
        values = get_rayleigh(x, scale)
        ax.fill_between(x + shift, values * y_scale, alpha=0.2, color = (colors[i + 1]))
        
    exp_xs = np.linspace(0, 9, 2000)
    
    line, = ax.plot(exp_xs, np.exp(-exp_xs * 0.3), color = colors[0], label = 'Transmittance')
    dashed_line = ax.axvline(x = -1, ymax = 1, linewidth=0.8, color='gray', linestyle = '--', alpha=1.0, label = 'Wavefront position')
    plots.extend([line, dashed_line])
    handler_map[dashed_line] = HandlerVerticalLine('grey')
    
    for i, (up_lim, scale, shift) in enumerate(zip(up_limits, scale_factors, shifts)):
        x = np.linspace(0, up_lim, 2000)
        values = get_rayleigh(x, scale)
        line, = ax.plot(x + shift, values * y_scale, label='Radiance distribution of $T_{res' + f'{i+1}' + '}$', color = colors[i + 1], linestyle = 'solid', linewidth = 1.5)
        ax.plot(np.full(2, shift), (0, 1), color = (colors[i + 1]), linestyle = '--', linewidth = 1.5, alpha = 0.7)
        draw_sample(ax, sample_position[i], colors[i + 1])

        plots.append(line) 
        handler_map[line] = HandlerRectangle(colors[i + 1], 0.2)
    # Customize the plot
    ax.set_xlim([0, 9.21])
    ax.set_ylim([-0.05, 1.17])
    
    ax.annotate("", xy=(9.2, 0), xytext=(-0.03, 0),
             arrowprops=dict(arrowstyle='->,head_length=0.5,head_width=0.2', color='black', fc='black'))

    # Draw arrow on the y-axis with a filled triangle
    ax.annotate("", xy=(0, 1.1), xytext=(0, -0.01),
                arrowprops=dict(arrowstyle='->,head_length=0.5,head_width=0.2', color='black', fc='black'))
    
    # Turn off the frame and ticks
    plt.legend(handles=plots, handler_map=handler_map)
    ax.axis('off')
    ax.grid(axis = 'x')

    # Show the plot
    plt.savefig('samples.png', dpi = 320)