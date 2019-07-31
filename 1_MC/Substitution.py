import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from numpy import linalg as LA
from collections import OrderedDict


def mutate(subs_freqs, nucl_ids, seq):
    new_seq = ''
    for nucl in seq:
        nucl_idx = np.random.choice(4, 1, p=subs_freqs[nucl_ids[nucl]])[0]
        new_seq += list(nucl_ids.keys())[nucl_idx]

    return new_seq


# Substitution
if __name__ == '__main__':

    nucleotides = OrderedDict()
    substitution_list = list()

    # Считаем матрицу замен
    with open('./data/substitution_matrix.txt') as freq_file:
        for idx, line in enumerate(freq_file):
            line = line.strip().split()
            nucleotide = line[0]
            substitution_list.append(list(map(float, line[1:])))
            nucleotides[nucleotide] = idx
    substitution_matrix = np.array(substitution_list)

    # Начальнвя строка
    initial = 'A' * 1000 + 'C' * 1000 + 'G' * 1000 + 'C' * 1000

    conv = []
    n_step = 20
    seq = initial
    for step in range(n_step):
        seq = mutate(substitution_matrix, nucleotides, seq)

        freq = Counter(seq)
        norm = sum(freq.values())
        simulated_freq = np.array([freq[nucl] / norm for nucl in nucleotides])

        w, v = LA.eig(substitution_matrix.T)
        j_stationary = np.argmin(abs(w - 1.0))
        p_stationary = v[:, j_stationary].real
        p_stationary /= p_stationary.sum()

        if step % (n_step // 10) == 0:
            print(simulated_freq)
            print(p_stationary)

        conv.append(np.linalg.norm(simulated_freq - p_stationary))

    plt.plot(np.array(conv))
    plt.show()
