# borrow from
# https://matplotlib.org/stable/gallery/axisartist/demo_axisline_style.html#sphx-glr-gallery-axisartist-demo-axisline-style-py
from mpl_toolkits.axisartist.axislines import AxesZero
import matplotlib.pyplot as plt
import numpy as np
import random

BUCKET_SIZE = 50
GAP = 1.0 / BUCKET_SIZE


def rand_zero_to_one():
    return random.uniform(0.0, 1.0)


def metropolis_samples(sampled_function, num_samples):

    def mutate(x):
        return rand_zero_to_one()

    buckets = {}
    for idx in range(BUCKET_SIZE):
        buckets[idx] = []

    def put_into_buckets(x, f_x):
        idx = int(x / GAP)
        buckets[idx].append(f_x)

    # to start markov chain
    # https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables#TheInversionMethod

    markov_start_candidates = []
    cdf = []
    # random sample 10000 candidates
    for _ in range(10000):
        x = rand_zero_to_one()
        weight = sampled_function(x) / 1.0
        markov_start_candidates.append((x, weight))
        cdf.append(weight)

    random_prob = rand_zero_to_one() * sum(cdf)
    candidate_id = len(cdf) - 1

    # inverse CDF
    for idx in range(len(cdf)):
        if random_prob <= cdf[idx]:
            candidate_id = idx
            break
        random_prob -= cdf[idx]

    markov_x0, _ = markov_start_candidates[candidate_id]
    global_weight = np.average(
        list(map(lambda x: x[1], markov_start_candidates)))

    print("markov_x0: {:.2f} -- weight: {:.3f}".format(markov_x0,
                                                       global_weight))
    # end of start-up

    # metropolis sampling:
    x = markov_x0
    for _ in range(num_samples):
        x_prime = mutate(x)
        f_x = black_box_function(x)
        f_x_prime = black_box_function(x_prime)
        prob_accept = min(1.0, f_x_prime / f_x)

        put_into_buckets(x, f_x * (1.0 - prob_accept) / global_weight)
        put_into_buckets(x_prime, f_x_prime * prob_accept / global_weight)

        if rand_zero_to_one() < prob_accept:
            x = x_prime

    return buckets


if __name__ == "__main__":
    for formula, black_box_function in [
        ("y = x", lambda x: x),
        ("y = x^2", lambda x: x * x),
        ("(x - 0.5)^2", lambda x: (x - 0.5)**2),
    ]:
        buckets = metropolis_samples(black_box_function, 100000)

        x_series = []
        y_series = []
        for idx in range(BUCKET_SIZE):
            if len(buckets[idx]) == 0:
                continue
            '''
            print("samples in [{:.3f}, {:.3f}): {}".format(
                idx * GAP, (idx + 0.5) * GAP, len(buckets[idx])))
            '''

            x = (idx + 0.5) * GAP
            y = np.average(buckets[idx])

            x_series.append(x)
            y_series.append(y)

            if idx == BUCKET_SIZE - 1:
                print("{:.2f} -> {:.3f}".format(x, y))

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

        ax.plot(x_series, y_series, label="metropolis sampling")
        #ax.set_aspect('equal', adjustable='box')

        t = np.arange(0., 1., 0.01)
        ax.plot(t, black_box_function(t), 'r--', label=formula)

        fig.legend(labelcolor="linecolor")

        file_name = "metropolis_samples_{}.png".format(formula.replace(" ", ""))

        plt.savefig(file_name, dpi=160)
        print("image saved to `{}`\n".format(file_name))
