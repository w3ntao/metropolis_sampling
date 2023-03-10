# python3 metropolis_sampling.py

import matplotlib.pyplot as plt
import numpy as np
import random

BUCKET_SIZE = 50
GAP = 1.0 / BUCKET_SIZE


def rand_zero_to_one():
    return random.uniform(0.0, 1.0)


def metropolis_sampling(black_box_function, num_samples):
    # to start markov chain
    # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Metropolis_Sampling#Start-upBias

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

    # TODO: how do I use `global_weight`?
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

    def record(x, f_x, weight):
        if weight <= 0.0:
            return

        buckets[int(x / GAP)].append((f_x, weight))

    # metropolis sampling:
    x = markov_x0
    for _ in range(num_samples):
        x_prime = mutate(x)
        f_x = black_box_function(x)
        f_x_prime = black_box_function(x_prime)
        prob_accept = min(1.0, f_x_prime / f_x)

        # 13.4.1 Basic Algorithm:
        # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Metropolis_Sampling#BasicAlgorithm
        record(x, f_x, 1.0 - prob_accept)
        record(x_prime, f_x_prime, prob_accept)

        if rand_zero_to_one() < prob_accept:
            x = x_prime

    return buckets


if __name__ == "__main__":
    for case_id, (formula, unknown_function) in enumerate([
        ("y = x", lambda x: x),
        ("y = x^2", lambda x: x * x),
        ("y = (x - 0.5)^2", lambda x: (x - 0.5)**2),
        ("y = 1.0 + sin(4*??*x)", lambda x: 1.0 + np.sin(x * 4 * np.pi)),
    ]):

        print("reconstructing `{}`".format(formula))

        num_samples = 5000
        buckets = metropolis_sampling(unknown_function, num_samples)

        samples_x = []
        samples_y = []

        pdf_samples_x = []
        pdf_samples_y = []

        for idx in range(BUCKET_SIZE):
            if len(buckets[idx]) == 0:
                continue

            accumulated_weight = 0.0
            accumulated_f_x = 0.0

            for f_x, weight in buckets[idx]:
                accumulated_weight += weight
                accumulated_f_x += f_x * weight

            samples_x.append((idx + 0.5) * GAP)
            samples_y.append(accumulated_f_x / accumulated_weight)

            pdf_samples_x.append((idx + 0.5) * GAP)
            pdf_samples_y.append(accumulated_weight / num_samples)

        fig = plt.figure()
        plt.subplot(121)

        t = np.arange(0.0, 1., 0.01)
        plt.plot(t,
                 unknown_function(t),
                 linestyle=':',
                 color="royalblue",
                 label="f: {}".format(formula))

        plt.plot(samples_x,
                 samples_y,
                 linestyle=":",
                 color='red',
                 label="reconstructed f")

        plt.subplot(122)

        plt.plot(pdf_samples_x,
                 pdf_samples_y,
                 color='magenta',
                 label="pdf of samples")

        fig.legend(labelcolor="linecolor",)

        file_name = "sampling_{}.png".format(case_id)

        plt.savefig(file_name, dpi=160, bbox_inches="tight")
        print("image saved to `{}`\n".format(file_name))
