import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import factorial


lmbd = 7.3
n_steps = 50


def suggest(k):
    # Самостоятельно


def prob(old_k, new_k):
    # Самостоятельно


def mcmc(init=7):
    k = init
    for i in range(n_steps):
        new_k = suggest(k)
        if random.random() < prob(k, new_k):
            k = new_k
    return k


if __name__ == '__main__':

    distr = []
    for j in range(10000):
        distr.append(mcmc())

    distr = np.array(distr)

    t = np.arange(0, 20, 0.1)
    d = np.exp(-lmbd) * np.power(lmbd, t) / factorial(t)

    plt.plot(t, d, '--')

    plt.hist(distr, normed=True)
    plt.show()
