import random
import math
import numpy as np
import matplotlib.pyplot as plt
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
    eps = 1
    n_steps = 1500

    distr = []
    for j in range(1000):
        distr.append(mcmc())

    distr = np.array(distr)

    mu = 0
    variance = eps
    sigma = math.sqrt(variance)
    x = np.linspace(0, 10, 100)
    plt.plot(x, 2 * norm.pdf(x, mu, sigma), '--')

    plt.hist(distr, normed=True)
    plt.show()
