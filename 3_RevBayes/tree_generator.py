import numpy as np
from random import choice
from random import randint
from collections import OrderedDict


def mutate(subs_freqs, nucl_ids, seq):
    idx = randint(1, len(seq) - 1)
    new_seq = seq[:idx]
    nucl = seq[idx]
    nucl_idx = np.random.choice(4, 1, p=subs_freqs[nucl_ids[nucl]])[0]
    new_seq += list(nucl_ids.keys())[nucl_idx]
    new_seq += seq[idx + 1:]

    return new_seq


if __name__ == '__main__':

    # Создадим случайную топологию
    tree = {
        1: {4},
        2: {4},
        3: {4},
        4: {1, 2, 3}
    }

    for _ in range(13):
        # take random branch
        v = choice(list(tree.keys()))
        u = choice(list(tree[v]))
        # append new branch
        new_vert = len(tree) + 1
        new_leaf = new_vert + 1
        tree[v] -= {u}
        tree[u] -= {v}
        tree[v].add(new_vert)
        tree[u].add(new_vert)
        tree[new_vert] = {v}
        tree[new_vert].add(u)
        tree[new_vert].add(new_leaf)
        tree[new_leaf] = {new_vert}

    # Сгенерируем длины ребер случайно, по Пуассону
    branch_lens_dict = dict()
    for u in tree:
        for v in tree[u]:
            branch_len = np.random.poisson(3, 1)[0] + 1
            branch_lens_dict[(u, v)] = branch_len

    # Пусть у нас будет какая нибудь простая матрица замен
    substitution_matrix = np.array(
        [
            [.5, .1, .2, .2],
            [.1, .5, .2, .2],
            [.2, .2, .5, .1],
            [.2, .2, .1, .5]
        ]
    )
    nucleotides = OrderedDict({'A': 0, 'C': 1, 'G': 2, 'T': 3})

    # Согласно топологии и длинам сгенерируем строки в узлах
    root_id = 4
    root = ''.join([choice('ATGC') for _ in range(250)])

    seq_in_verts = {k: '' for k in tree}
    seq_in_verts[root_id] = root
    visited = set()


    def dfs(u):
        visited.add(u)
        for v in tree[u]:
            if v not in visited:
                n_steps = branch_lens_dict[(u, v)]
                src_seq = seq_in_verts[u]
                for _ in range(n_steps):
                    src_seq = mutate(substitution_matrix, nucleotides, src_seq)
                seq_in_verts[v] = src_seq
                dfs(v)

    dfs(root_id)

    # Запишем послодовательности получившихся таксонов
    final_sequences = []
    for vert in sorted(seq_in_verts.keys()):
        if len(tree[vert]) == 1:
            print(vert, seq_in_verts[vert])
            final_sequences.append((vert, seq_in_verts[vert]))

    # Сохраним в NEXUS файл
    head = '''#NEXUS

BEGIN DATA;
DIMENSIONS  NTAX={} NCHAR={};
FORMAT DATATYPE=DNA GAP=- MISSING=?;
MATRIX

'''
    with open('./data/taxa.nex', 'w') as taxa_file:
        taxa_file.write(
            head.format(
                str(len(final_sequences)),
                str(len(root))
            )
        )
        idx = 1
        for taxa_idx, seq in final_sequences:
            taxa_file.write(str(idx) + '\n')
            idx += 1
            taxa_file.write(seq + '\n')
        taxa_file.write(';\n\nEND;')
