from mpl_toolkits.axisartist.axislines import AxesZero
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(axes_class=AxesZero)

for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)


def gt_samples():
    x = np.linspace(0.0, 1.0, 100)
    return x, (x - 0.5)**2


x_series, y_series = gt_samples()
ax.plot(x_series, y_series)
#ax.set_aspect('equal', adjustable='box')

plt.savefig('foo.png')
