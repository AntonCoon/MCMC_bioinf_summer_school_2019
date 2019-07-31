import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import norm


def suggest(x):
    # Самостоятельно


def prob(old_x, new_x):
    # Самостоятельно


def mcmc(init=0):
    x = init
    for i in range(n_steps):
        new_x = suggest(x)
        if random.random() < prob(x, new_x):
            x = new_x
    return x


if __name__ == '__main__':
    eps = .1
    n_steps = 1500

    distr = []
    for j in range(1000):
        distr.append(mcmc())

    distr = np.array(distr)

    x = np.linspace(expon.ppf(0.1), expon.ppf(0.99), 100)
    plt.plot(x, expon.pdf(x), '--', label='exponential pdf')

    plt.hist(distr, normed=True)
    plt.show()
