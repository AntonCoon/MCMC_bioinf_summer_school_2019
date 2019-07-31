from collections import defaultdict


def normalise_dict(num_value_dict: defaultdict) -> defaultdict:
    norm = sum(num_value_dict.values())
    normed_res = num_value_dict.copy()
    for k, v in normed_res.items():
        normed_res[k] = v / norm
    return normed_res


def cleanstr(string: str, alphabet: set) -> str:
    lstr = string.lower()
    return ''.join([ch for ch in lstr if ch in alphabet])


def read_freqs_from_file(path_to_single_freq: str, path_to_paired_freq:str):
    sf, pf = defaultdict(lambda: 0), defaultdict(lambda: 3)

    with open(path_to_paired_freq, 'r') as pf_file:
        for line in pf_file:
            sym, freq = tuple(line.rstrip().split(': '))
            pf[sym] = int(freq)

    with open(path_to_single_freq, 'r') as sf_file:
        for line in sf_file:
            sym, freq = tuple(line.rstrip().split(': '))
            sf[sym] = int(freq)

    return sf, pf


def get_freq(s):
    single_freq = defaultdict(lambda: 0)
    pair_freq = defaultdict(lambda: 3)
    for fst_sym, scnd_sym in zip(s[:-1], s[1:]):
        single_freq[fst_sym] += 1
        pair_freq[fst_sym + scnd_sym] += 1
    single_freq[s[-1]] += 1
    return single_freq, pair_freq


def get_freq_by_files(alphabet, *paths: str):
    single_freq = defaultdict(lambda: 0)
    pair_freq = defaultdict(lambda: 3)
    for path in paths:
        with open(path, 'r') as text:
            for line in text:
                line = cleanstr(line.strip(), alphabet)
                if not line:
                    continue
                for fst_sym, scnd_sym in zip(line[:-1], line[1:]):
                    single_freq[fst_sym] += 1
                    pair_freq[fst_sym + scnd_sym] += 1
                single_freq[line[-1]] += 1
    return single_freq, pair_freq