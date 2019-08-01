import matplotlib.pyplot as plt
import numpy as np
import distance
import networkx as nx
import pandas as pd
import seaborn as sns

from Bio import Phylo


def get_matrix_by_string(strings: list) -> np.array:
    n_row = len(strings)
    distances_matrix = [[0] * n_row for _ in range(n_row)]

    for row in range(n_row):
        for col in range(row):
            dist = distance.hamming(strings[row], strings[col])
            distances_matrix[row][col] = dist
            distances_matrix[col][row] = dist

    return np.array(distances_matrix)


def get_tree_and_leafs(path_to_nex: str) -> (nx.Graph, list):
    def inc(i):
        i[0] += 1
        return i[0]

    phylo_tree = Phylo.read(path_to_nex, 'nexus')
    net_tree = Phylo.to_networkx(phylo_tree)

    taxa_n = phylo_tree.count_terminals()

    tree = nx.Graph()
    idx = [0]
    for edge in nx.dfs_edges(net_tree):
        u, v = tuple(edge)
        u.name = str(inc(idx) + taxa_n) if u.name is None else u.name
        v.name = str(inc(idx) + taxa_n) if v.name is None else v.name
        len_u = u.branch_length
        len_v = v.branch_length
        len_u = len_u if type(len_u) is float else float(len_u.split('[')[0])
        len_v = len_v if type(len_v) is float else float(len_v.split('[')[0])
        edg_len = abs(len_u - len_v)
        tree.add_edge(u.name, v.name, length=edg_len)

    leafs = [x for x in tree.nodes() if tree.degree(x) == 1]

    return tree, leafs


def get_matrix_by_tree(tree, leafs) -> np.array:
    leafs = sorted(leafs, key=int)
    n_row = len(leafs)
    distances_matrix = [[0] * n_row for _ in range(n_row)]

    for row in range(n_row):
        for col in range(row):
            dist = nx.dijkstra_path_length(tree, leafs[row], leafs[col],
                                           weight='length')
            distances_matrix[row][col] = dist
            distances_matrix[col][row] = dist

    return np.array(distances_matrix)


if __name__ == '__main__':

    simulated_taxa = []

    with open('./data/taxa.nex', 'r') as simulated:
        for idx, line in enumerate(simulated):
            if idx > 6 and idx % 2 == 0:
                simulated_taxa.append(line.strip())

    simulated_taxa = simulated_taxa[:-1]

    dist_by_original_str = get_matrix_by_string(simulated_taxa)

    np_str_dists = np.array([lst[:-1] for lst in dist_by_original_str[:-1]])
    mask = np.zeros_like(np_str_dists)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        heat_1 = sns.heatmap(
            pd.DataFrame(
                np_str_dists
            ),
            cmap='YlGnBu',
            annot=True,
            cbar=False,
            square=True,
            mask=mask
        )
        plt.savefig('./output/dist_by_original_str.pdf')
        plt.show()

    rec_tree, leafs = get_tree_and_leafs('./output/gama_map_tree.tree')
    dist_by_assembled_tree = get_matrix_by_tree(rec_tree, leafs)

    np_tree_dists = np.array([lst[:-1] for lst in dist_by_assembled_tree[:-1]])
    mask = np.zeros_like(np_tree_dists)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        heat_2 = sns.heatmap(
            pd.DataFrame(
                np_tree_dists * 100
            ),
            cmap='YlGnBu',
            annot=True,
            cbar=False,
            square=True,
            mask=mask
        )
        plt.savefig('./output/dist_by_assembled_tree.pdf')
        plt.show()

    dst_str = []
    for i in range(len(np_str_dists) - 1):
        for j in range(i):
            dst_str.append(np_str_dists[i, j])
    dst_str = np.array(dst_str)
    # dst_str = np.resize(np_str_dists, np_str_dists.size)
    # dst_mcmc_tree = np.resize(np_tree_dists, np_str_dists.size)
    dst_mcmc_tree = []
    for i in range(len(np_tree_dists) - 1):
        for j in range(i):
            dst_mcmc_tree.append(np_tree_dists[i, j])
    dst_mcmc_tree = np.array(dst_mcmc_tree)
    ax = plt.plot(
        dst_str,
        dst_mcmc_tree,
        'o'
    )
    plt.title('corr = {}'.format(np.corrcoef(dst_str, dst_mcmc_tree)[0, 1]))
    plt.savefig('./output/corr_plot.pdf')
    plt.show()
