import itertools
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from config import (CODEBOOK, DERIVED, RESULTS, MIN_CODE_COUNT, MIN_COUNT, MIN_NPMI,
                    MIN_EXPECTED, MAX_NPMI, SAMPLE_FILTER)
from ocm_graph_builder import build_graph

def count_pairs(code_lists):
    cooccurence_table = Counter()
    code_counts = Counter()  # number of paragraphs each code appears in
    for codes in code_lists:
        unique_codes = sorted(set(codes))
        code_counts.update(unique_codes)
        for pair in itertools.combinations(unique_codes, 2):
            cooccurence_table[pair] += 1
    return cooccurence_table, code_counts


def compute_associations(pair_counts, code_counts, total_records, min_code_count):
    """
    Computes Lift, PMI, and NPMI from a Counter of pair co-occurrences and a
    Counter of per-code counts, keeping only pairs where both codes appear in
    at least min_code_count paragraphs. inlcuding ones that never co-ccur.
    total_records is the number of paragraphs the counts were taken from.
    """

    frequent_codes = sorted(code for code, count in code_counts.items() if count >= min_code_count)
    pairs = pd.DataFrame(
        [(a, b, pair_counts[(a, b)]) for a, b in itertools.combinations(frequent_codes, 2)],
        columns=['code_A', 'code_B', 'count_AB']
    )

    if pairs.empty:
        return pd.DataFrame() # Return empty if no pairs pass the threshold

    pairs['P_AB'] = pairs['count_AB'] / total_records

    #share of paragraphs containing each code
    pairs['P_A'] = pairs['code_A'].map(code_counts) / total_records
    pairs['P_B'] = pairs['code_B'].map(code_counts) / total_records
    pairs['expected_AB'] = pairs['P_A'] * pairs['P_B'] * total_records

    #Lift, PMI, and NPMI
    pairs['lift'] = pairs['P_AB'] / (pairs['P_A'] * pairs['P_B'])

    with np.errstate(divide='ignore', invalid='ignore'):                  # log2(0) is expected here
        pairs['pmi'] = np.log2(pairs['lift'])
        neg_log_p_ab = -np.log2(pairs['P_AB'])
        pairs['npmi'] = np.select(
            [pairs['count_AB'] == 0, neg_log_p_ab == 0],   # never together / always together
            [-1.0, 1.0],
            default=pairs['pmi'] / neg_log_p_ab)

    # Clean up and sort by strength of association
    result = pairs[['code_A', 'code_B', 'count_AB', 'P_AB', 'expected_AB', 'lift', 'pmi', 'npmi']]
    return result.sort_values(by='npmi', ascending=False).reset_index(drop=True)

def cooccurence_graph_pos(G, associations, min_npmi=0.3, min_count=2):
    cooccur_graph_pos = nx.Graph()
    cooccur_graph_pos.add_nodes_from((n, d) for n, d in G.nodes(data=True) if d.get("level") != "major")
    for row in associations.itertuples(index=False):
        if row.npmi >= min_npmi and row.count_AB >= min_count:
            cooccur_graph_pos.add_edge(row.code_A, row.code_B, kind="cooccur",
                                   weight=row.npmi, instances=row.count_AB)
    return cooccur_graph_pos

def cooccurence_graph_neg(G, associations, max_npmi=-0.3, min_expected=5):
    cooccur_graph_neg = nx.Graph()
    cooccur_graph_neg.add_nodes_from((n, d) for n, d in G.nodes(data=True) if d.get("level") != "major")
    for row in associations.itertuples(index=False):
        if row.npmi <= max_npmi and row.expected_AB >= min_expected:
            cooccur_graph_neg.add_edge(row.code_A, row.code_B, kind="avoid",
                                       weight=-row.npmi, expected=row.expected_AB,
                                       instances=row.count_AB)
    return cooccur_graph_neg


if __name__ == "__main__":
    codes_derived = pd.read_parquet(DERIVED / "codes.parquet")
    tag = "all"
    if SAMPLE_FILTER:
        codes_derived = codes_derived[codes_derived['samples'].fillna('').str.contains(SAMPLE_FILTER)]
        tag = SAMPLE_FILTER

    pair_counts, code_counts = count_pairs(codes_derived['ocms'])
    associations = compute_associations(pair_counts, code_counts, len(codes_derived), MIN_CODE_COUNT)

    wrdnet = build_graph(CODEBOOK)
    pos_network = cooccurence_graph_pos(wrdnet, associations, MIN_NPMI, MIN_COUNT)
    neg_network = cooccurence_graph_neg(wrdnet, associations, MAX_NPMI, MIN_EXPECTED)

    RESULTS.mkdir(exist_ok=True)
    associations.to_csv(RESULTS / f"associations_{tag}.csv", index=False)
    nx.write_gexf(pos_network, RESULTS / f"cooccur_pos_{tag}.gexf")
    nx.write_gexf(neg_network, RESULTS / f"cooccur_neg_{tag}.gexf")

    print(f"{len(codes_derived):,} paragraphs, {len(associations):,} pairs of frequent codes")
    print(f"positive edges: {pos_network.number_of_edges()}, negative edges: {neg_network.number_of_edges()}")
    print(f"results saved to {RESULTS}/ with tag '{tag}'")