import matplotlib.pyplot as plt

# Your plotting code
plt.plot([1, 2, 3], [4, 5, 6])

# Draw arrow on the x-axis with a filled triangle
plt.annotate("", xy=(1.2, 4), xytext=(1.0, 4),
             arrowprops=dict(arrowstyle='->,head_length=0.5,head_width=0.2', color='black', fc='black'))

# Draw arrow on the y-axis with a filled triangle
plt.annotate("", xy=(1, 4.5), xytext=(1, 4),
             arrowprops=dict(arrowstyle='->,head_length=0.5,head_width=0.2', color='black', fc='black'))

# Turn off the frame and ticks
# plt.axis('off')

# Show the plot
plt.show()