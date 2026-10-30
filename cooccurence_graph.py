import pandas as pd
import numpy as np
import itertools
import networkx as nx
import pyarrow
from collections import Counter
from ocm_graph_builder import build_graph


#file path and important columns 'ocms', 'type', 'pageEid', 'sreid','sreprev', 'srenext', 'parent', 'section', 'sectpar', 'sectgpar', 'division','coverage', 'hdoc', 'owcs', 'mainowc', 'samples' 
file_path = 'sample/fake_ethnotext.csv.gz'
selected_columns = ['ocms', 'type', 'sreid', 'hdoc', 'mainowc']
chunk_size = 100_000


# Iterate in chunks
chunk_iterator = pd.read_csv(
    file_path, 
    usecols=selected_columns, 
    chunksize=chunk_size
)

chunk_list = []

#build code only derived table, filter for non p, #000, code appearence min, 
for chunk in chunk_iterator:
    chunk = chunk[(chunk['type'] == 'p')].copy()
    chunk['ocms'] = chunk['ocms'].fillna('').str.split().apply(lambda x: [i[1:] for i in x if i != '#000'])
    chunk_list.append(chunk)

codes_derived = pd.concat(chunk_list, ignore_index=True)
codes_derived = codes_derived[codes_derived['ocms'].str.len() > 0]

#build table
cooccurence_table = Counter()
code_counts = Counter()  # number of paragraphs each code appears in
for codes in codes_derived['ocms']:
    unique_codes = sorted(set(codes))
    code_counts.update(unique_codes)
    for pair in itertools.combinations(unique_codes, 2):
        cooccurence_table[pair] += 1


def compute_associations(pair_counts, code_counts, total_records, min_code_count):
    """
    Computes Lift, PMI, and NPMI from a Counter of pair co-occurrences and a
    Counter of per-code counts, keeping only pairs where both codes appear in
    at least min_code_count paragraphs.
    total_records is the number of paragraphs the counts were taken from.
    """
    # 1. Keep only codes that appear often enough
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

associations = compute_associations(cooccurence_table, code_counts, len(codes_derived), 5)

wrdnet = build_graph("ocmdefs.txt")

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

pos_network = cooccurence_graph_pos(wrdnet, associations)
neg_network = cooccurence_graph_neg(wrdnet, associations, min_expected=0.1)  # sample only: max expected_AB is ~0.2; use the default 5 on the full data

#display cooccurence_graph
associations.to_csv("outputs/associations.csv")
nx.write_gexf(pos_network, "outputs/cooccur_pos.gexf")
nx.write_gexf(neg_network, "outputs/cooccur_neg.gexf")