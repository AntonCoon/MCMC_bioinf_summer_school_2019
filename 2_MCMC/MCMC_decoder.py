import math
import random
import numpy
import matplotlib.pyplot as plt
from itertools import product
from funcs import *


def log_like(message, pf):
    res = 0
    n = len(message)
    for i in range(n - 1):
        res += math.log(pf[message[i] + message[i + 1]])
    return res


def major_freq_decoder(
        message_single_freq: defaultdict,
        corpus_single_freq: defaultdict) -> dict:
    code_left, code_right = [], []
    for sym in alphabet:
        code_right.append((sym, message_single_freq[sym]))
        code_left.append((sym, corpus_single_freq[sym]))
    code_left = sorted(code_left, key=lambda pair: pair[1])
    code_right = sorted(code_right, key=lambda pair: pair[1])
    return {k: v for (k, _), (v, _) in zip(code_left, code_right)}


def code_message(message, code):
    res = ''
    for letter in message:
        res += code[letter]
    return res


def suggest(code, alphabet):
    new_code = code.copy()
    a, b = random.sample(alphabet, 2)
    code[a], code[b] = code[b], code[a]
    return new_code


def suggest_modified(code, alphabet):
    new_code = code.copy()
    m = random.randint(2, 17)
    shaked = random.sample(alphabet, m)
    perm = numpy.random.permutation(shaked)
    tuple2 = []
    for i in range(len(shaked)):
        tuple2.append(code[perm[i]])
    tuple2 = tuple(tuple2)
    for i in range(len(shaked)):
        code[shaked[i]] = tuple2[i]
    return new_code


def prob(old_code, new_code, message, paired_freqs):
    res = 1
    for fst_sym, scnd_sym in zip(message[:-1], message[1:]):
        new_freq = paired_freqs[new_code[fst_sym] + new_code[scnd_sym]]
        old_freq = paired_freqs[old_code[fst_sym] + old_code[scnd_sym]]
        res *= (new_freq / old_freq)
    return res


def mcmc(init_code, message, paired_freq: defaultdict, alphabet: set, n_steps):
    conv = list()
    code = init_code.copy()
    for step in range(n_steps):
        new_code = suggest_modified(code, alphabet)

        if random.random() < prob(code, new_code, message, paired_freq) ** .1:
            code = new_code
            conv.append(log_like(code_message(message, code), paired_freq))
            if step % (n_steps // 10) == 0:
                print(
                    'Step #{}:\n{}'.format(
                        str(step), code_message(message, code)
                    )
                )
    return code, conv


if __name__ == '__main__':

    alphabet = [chr(idx) for idx in range(1072, 1072 + 32)] + ['ё', ' ']
    alphabet_set = set(alphabet)

    sf, pf = get_freq_by_files(alphabet_set, './data/corp.txt')
    for sym_1, sym_2 in product(alphabet, alphabet):
        pf[sym_1 + sym_2] += 0
    pf = normalise_dict(pf)

    message_id = 2
    # Read original message for estimations
    original_messages = []
    with open('./data/original_messages.txt', 'r') as message_file:
        for line in message_file:
            original_messages += [line.strip()]
    original_message = original_messages[message_id]

    # Read encoded message
    encoded_messages = []
    with open('./data/encoded_messages.txt', 'r') as encoded_message_file:
        for line in encoded_message_file:
            encoded_messages += [line.strip()]
    encoded_message = encoded_messages[message_id]

    # Count letters and their pairs frequency in encoded message
    sfm, pfm = get_freq(encoded_message)

    # Simple decoding by frequency
    simple_code = major_freq_decoder(sfm, sf)
    print(
        'Message decoded by frequency:\n{}'.format(
            code_message(encoded_message, simple_code)
        )
    )

    # MCMC decoding
    final_code, convergate = mcmc(
        simple_code,
        encoded_message,
        pf,
        alphabet_set,
        n_steps=100000
    )

    reconstructed_message = code_message(encoded_message, final_code)
    print('Final message decoded with MCMC:\n{}'.format(reconstructed_message))

    original_like = log_like(original_message, pf)
    plt.plot(numpy.array(convergate), '-')
    plt.plot(numpy.array([original_like for _ in range(len(convergate))]), '-')
    plt.savefig('./out/convergate.pdf')
    plt.show()
