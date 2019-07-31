import random
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats.distributions import poisson
import numpy as np


def suggest(k):
    if random.random() < 0.5:
        return k - 1
    else:
        return k + 1


def prob(old_k, new_k):
    # Самостоятельно
    if new_k < 0:
        return 0
    if new_k > old_k:
        return lmbd / new_k
    else:
        return old_k / lmbd


def mcmc(init=7):
    k = init
    res = []
    for i in range(n_steps):
        new_k = suggest(k)
        if random.random() < prob(k, new_k):
            k = new_k
        res.append(k)
    return res


if __name__ == '__main__':
    lmbd = 7.3
    n_steps = 20

    distr = []
    for j in range(1000):
        distr.append(mcmc())

    print(', '.join(map(str, distr[-1])))

    conv = []
    freq = []
    for idx in range(n_steps):
        distr_on_step = [line[idx] for line in distr]
        # print(distr_on_step)
        freq = Counter(distr_on_step)
        norm = sum(freq.values())
        diff = 0
        for num in range(0, 20):
            if num in freq:
                diff += abs(freq[num] / norm - poisson.pmf(num, lmbd))
        conv.append(diff)

    plt.plot(np.array(conv))
    plt.show()
