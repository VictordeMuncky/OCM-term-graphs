import pandas as pd
import numpy as np
import itertools
from collections import Counter


#file path and important columns
file_path = 'sample/fake_ethnotext.csv.gz'
selected_columns = ['ocms', 'type', 'pageEid', 'sreid',
                    'sreprev', 'srenext', 'parent', 'section', 'sectpar', 'sectgpar', 'division',
                     'coverage', 'hdoc', 'owcs', 'mainowc', 'samples']
chunk_size = 100


# Iterate in chunks
chunk_iterator = pd.read_csv(
    file_path, 
    usecols=selected_columns, 
    chunksize=chunk_size
)

chunk_list = []

#build code only derived table, filter for non p, #000, code appearence min, 
for chunk in chunk_iterator:
    chunk = chunk[(chunk['type'] == 'p')]
    chunk['ocms'] = chunk['ocms'].fillna('').str.split(' ').apply(lambda x: [i[1:] for i in x if i != '#000'])
    chunk_list.append(chunk)

codes_derived = pd.concat(chunk_list, ignore_index=True)

#build table
cooccurence_table = Counter()
code_counts = Counter()  # number of paragraphs each code appears in
for codes in codes_derived['ocms']:
    unique_codes = sorted(set(codes))
    code_counts.update(unique_codes)
    for pair in itertools.combinations(unique_codes, 2):
        cooccurence_table[pair] += 1


def compute_associations(pair_counts, code_counts, total_records, min_support=0.005):
    """
    Computes Lift, PMI, and NPMI from a Counter of pair co-occurrences and a
    Counter of per-code counts, pruning pairs below the min_support threshold.
    total_records is the number of paragraphs the counts were taken from.
    """
    # 1. Joint counts per pair, filtered by minimum support
    pairs = pd.DataFrame(
        [(a, b, count) for (a, b), count in pair_counts.items()],
        columns=['code_A', 'code_B', 'count_AB']
    )
    pairs['P_AB'] = pairs['count_AB'] / total_records
    pairs = pairs[pairs['P_AB'] >= min_support].copy()

    if pairs.empty:
        return pd.DataFrame() # Return empty if no pairs pass the threshold

    # 2. Marginal probabilities: share of paragraphs containing each code
    pairs['P_A'] = pairs['code_A'].map(code_counts) / total_records
    pairs['P_B'] = pairs['code_B'].map(code_counts) / total_records

    # 3. Lift, PMI, and NPMI
    pairs['lift'] = pairs['P_AB'] / (pairs['P_A'] * pairs['P_B'])
    pairs['pmi'] = np.log2(pairs['lift'])
    # NPMI divides by -log2(P_AB), which is 0 when a pair is in every paragraph; that case is perfect association (1)
    neg_log_p_ab = -np.log2(pairs['P_AB'])
    pairs['npmi'] = np.where(neg_log_p_ab > 0, pairs['pmi'] / neg_log_p_ab.where(neg_log_p_ab > 0), 1.0)

    # Clean up and sort by strength of association
    result = pairs[['code_A', 'code_B', 'count_AB', 'P_AB', 'lift', 'pmi', 'npmi']]
    return result.sort_values(by='npmi', ascending=False).reset_index(drop=True)

associations = compute_associations(cooccurence_table, code_counts, len(codes_derived))
print(associations)

#& ~chunk['ocms'].str.contains('#000', regex=False, na=False) add somewhere