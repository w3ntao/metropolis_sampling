# taken from
# https://matplotlib.org/stable/gallery/axisartist/demo_axisline_style.html#sphx-glr-gallery-axisartist-demo-axisline-style-py
from mpl_toolkits.axisartist.axislines import AxesZero
import matplotlib.pyplot as plt
import numpy as np
import random

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


def rand_float():
    return random.uniform(0.0, 1.0)


def metropolis_samples():

    def f(x):
        return x

    def mutate(x):
        return rand_float()

    BUCKET_SIZE = 10
    GAP = 1.0 / BUCKET_SIZE

    buckets = {}
    for idx in range(BUCKET_SIZE):
        buckets[idx] = []

    def put_into_buckets(x, f_x):
        idx = int(x / GAP)
        buckets[idx].append(f_x)

    # to start markov chain
    '''
    # TODO: implement me
    markov_start_x_candidates = []
    for _ in range(10):
        x = rand_float()
        markov_start_x_candidates.append(x, f(x) / 1.0)

    sorted(markov_start_x_candidates,)
    '''

    # end of start-up

    x0 = rand_float()
    f_x0 = f(x0)
    global_weight = f_x0
    print("x0: {:.2f} -- weight: {:.3f}\n".format(x0, global_weight))

    put_into_buckets(x0, f_x0 / global_weight)

    x = x0
    for _ in range(1000):
        x_prime = mutate(x)
        f_x = f(x)
        f_x_prime = f(x_prime)
        prob_accept = min(1.0, f_x_prime / f_x)

        put_into_buckets(x, f_x * (1.0 - prob_accept) / global_weight)
        put_into_buckets(x_prime, f_x_prime * prob_accept / global_weight)

        if rand_float() < prob_accept:
            x = x_prime

    x_series = []
    y_series = []
    for idx in range(BUCKET_SIZE):
        if len(buckets[idx]) == 0:
            continue
        #x_series.append((idx + 0.5) * GAP)

        x = idx * GAP
        y = np.average(buckets[idx])

        x_series.append(x)
        y_series.append(y)

        print("{:.2f} -> {:.3f}".format(x, y))

    return x_series, y_series


x_series, y_series = metropolis_samples()

ax.plot(x_series, y_series)
#ax.set_aspect('equal', adjustable='box')

file_name = "metropolis_samples.png"

plt.savefig(file_name)
print("\nimage saved to `{}`".format(file_name))

exit(0)

x_series, y_series = gt_samples()
ax.plot(x_series, y_series)
#ax.set_aspect('equal', adjustable='box')

plt.savefig('foo.png')
