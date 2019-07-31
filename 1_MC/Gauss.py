import random
from collections import Counter
import math


def suggest(x):
    return x + eps * (2 * random.random() - 1)


def prob(old_x, new_x):
    # Самостоятельно


def mcmc(init=0):
    x = init
    res = []
    for i in range(n_steps):
        new_x = suggest(x)
        if random.random() < prob(x, new_x):
            x = new_x
        res.append(x)
    return res


if __name__ == '__main__':
    eps = 0.1
    n_steps = 5000

    distr = []
    for j in range(1000):
        distr.append(mcmc())

    conv = []
    freq = []
    for idx in range(n_steps):
        distr_on_step = [line[idx] for line in distr]
        freq = Counter(distr_on_step)

    print(distr_on_step)
