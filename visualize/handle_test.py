import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Rectangle

# Create a custom Handler class for rectangles in the legend
class HandlerRectangle(HandlerPatch):
    def __init__(self, color, fill_alpha):
        super().__init__(self)
        self.color = color
        self.alpha = fill_alpha
        
    def create_artists(self, _1, orig_handle, _2, _3, width, height, _4, trans):
        w = 20                          # Width of the rectangle
        h = height
        line = Rectangle(xy=(0, h), width=w, height=0.1, linewidth=orig_handle.get_linewidth(), edgecolor = self.color, label=orig_handle.get_label())
        rect = Rectangle(xy=(0, 0), width=w, height=h, facecolor = self.color, 
                edgecolor = (0, 0, 0, 0), linewidth=0, label=orig_handle.get_label(), alpha = self.alpha)
        line.set_transform(trans)
        rect.set_transform(trans)
        return [line, rect]

# Your plotting code
line1, = plt.plot([1, 2, 3], [4, 5, 6], label='My Line1', color = 'red')
line2, = plt.plot([-1, -2, -3], [4, 5, 6], label='My Line2', color = 'blue')
line3, = plt.plot([0, 0, 0], [4, 5, 6], label='My Line3', color = 'green')

# Create a legend with a custom Handler
plt.legend(handles=[line1, line2, line3], handler_map={line1: HandlerRectangle('red', 0.2), line2: HandlerRectangle('blue', 0.2)})

# Customize the plot
plt.title('Plot with Custom Legend Entry')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()