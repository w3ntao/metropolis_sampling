# python3 metropolis_sampling.py

import matplotlib.pyplot as plt
import numpy as np
import random
import math

BUCKET_SIZE = 50
GAP = 1.0 / BUCKET_SIZE


def rand_zero_to_one():
    return random.uniform(0.0, 1.0)


def metropolis_sampling(black_box_function, num_samples):

    # to start markov chain
    # https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables#TheInversionMethod

    markov_start_candidates = []
    cdf = []
    # random sample 10000 candidates
    for _ in range(10000):
        x = rand_zero_to_one()
        weight = black_box_function(x) / 1.0
        markov_start_candidates.append((x, weight))
        cdf.append(weight)

    markov_start_candidates = sorted(markov_start_candidates,
                                     reverse=True,
                                     key=lambda x: x[1])

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

    def mutate(x):
        return rand_zero_to_one()

    buckets = {}
    for idx in range(BUCKET_SIZE):
        buckets[idx] = []

    def record(x, weight, f_x):
        if weight <= 0.0:
            return

        buckets[int(x / GAP)].append((x, weight, f_x))

    # metropolis sampling:
    x = markov_x0
    for _ in range(num_samples):
        x_prime = mutate(x)
        f_x = black_box_function(x)
        f_x_prime = black_box_function(x_prime)
        prob_accept = min(1.0, f_x_prime / f_x)

        # 13.4.1 Basic Algorithm:
        # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Metropolis_Sampling#BasicAlgorithm
        record(x, 1.0 - prob_accept, f_x)
        record(x_prime, prob_accept, f_x_prime)

        #record(x_prime, 1.0, f_x_prime)

        if rand_zero_to_one() < prob_accept:
            x = x_prime

    return buckets


if __name__ == "__main__":
    for formula, unknown_function in [
        ("y = x", lambda x: x),
        ("y = x^2", lambda x: x * x),
        ("(x - 0.5)^2", lambda x: (x - 0.5)**2),
        ("1.0 + 2*PI*x", lambda x: 1.0 + np.cos(x * 2 * np.pi)),
    ]:
        buckets = metropolis_sampling(unknown_function, 5000)

        x_series = []
        y_series = []
        for idx in range(BUCKET_SIZE):
            if len(buckets[idx]) == 0:
                continue

            total_weight = 0.0
            total_f_x = 0.0

            for x, weight, f_x in buckets[idx]:
                total_weight += weight
                total_f_x += f_x * weight

            x_series.append((idx + 0.5) * GAP)
            y_series.append(total_f_x / total_weight)

        fig = plt.figure()
        plt.subplot(121)

        t = np.arange(0.0, 1., 0.01)
        plt.plot(t,
                 unknown_function(t),
                 linestyle='--',
                 color="royalblue",
                 label="f: {}".format(formula))

        plt.plot(x_series,
                 y_series,
                 linestyle=":",
                 color='red',
                 label="reconstructed f")

        plt.subplot(122)
        num_samples_x = []
        num_samples_y = []
        for idx in range(BUCKET_SIZE):
            num_samples_x += [(idx + 0.5) * GAP]
            total_weight = sum(map(lambda item: item[1], buckets[idx]))
            num_samples_y += [total_weight]

        plt.plot(num_samples_x,
                 num_samples_y,
                 color='magenta',
                 label="number of samples")

        fig.legend(labelcolor="linecolor",)

        file_name = "metropolis_samples_{}.png".format(formula.replace(" ", ""))

        plt.savefig(file_name, dpi=160, bbox_inches="tight")
        print("image saved to `{}`\n".format(file_name))
