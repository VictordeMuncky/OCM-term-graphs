import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
import umap
import plotly.express as px

def ppmi_matrix(associations, filter=None):
    if filter is not None:
        drop = associations['code_A'].isin(filter) | associations['code_B'].isin(filter)
        associations = associations[~drop]

    all_items = sorted(set(associations['code_A']) | set(associations['code_B']))
    index = {code: i for i, code in enumerate(all_items)}

    rows = associations['code_A'].map(index).to_numpy()
    cols = associations['code_B'].map(index).to_numpy()
    ppmi = np.clip(associations['pmi'].to_numpy(), 0, None)

    matrix = np.zeros((len(all_items), len(all_items)))
    matrix[rows, cols] = ppmi
    matrix[cols, rows] = ppmi  # pairs are stored once (A < B), so mirror them

    return pd.DataFrame(matrix, index=all_items, columns=all_items)

def truncated_SVD(matrix, k=25):
    """
    Compresses the code x code PPMI matrix into a k-dimensional vector per code.
    Codes that co-occur with similar other codes end up with similar vectors.
    Returns a DataFrame with one row per code and k columns (dim_0 ... dim_{k-1}).
    """

    values = matrix.to_numpy(dtype=float)


    U, S, Vt = svds(values, k=k)

    # svds returns the patterns from weakest to strongest; flip so dim_0 is the strongest
    order = np.argsort(S)[::-1]
    U, S = U[:, order], S[order]

    vectors = U * np.sqrt(S)

    return pd.DataFrame(vectors, index=matrix.index,
                        columns=[f'dim_{i}' for i in range(k)])

def nearest_codes(code, k, truncated_matrix):
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    knn.fit(truncated_matrix)

    target_vector = truncated_matrix.loc[[code]]

    distances, indices = knn.kneighbors(target_vector)
    
    nearest_indices = indices[0][1:]
    codes = truncated_matrix.index[nearest_indices].tolist()

    return codes, distances[0][1:]

def dimension_poles(vectors, n=10):
    """
    For every SVD dimension, lists the n codes with the highest and the n codes
    with the lowest value on it. labels is an optional {code: name} dict, e.g. from
    the codebook graph, so the names are shown next to the codes.
    Returns a long DataFrame: dimension, pole ('high'/'low'), rank, code, label, value.
    """
    rows = []
    for dim in vectors.columns:
        for pole, picked in [('high', vectors[dim].nlargest(n)), ('low', vectors[dim].nsmallest(n))]:
            for rank, (code, value) in enumerate(picked.items(), start=1):
                rows.append((dim, pole, rank, code, value))
    return pd.DataFrame(rows, columns=['dimension', 'pole', 'rank', 'code', 'value'])

def umap_reduction(vectors, neighbours = 10, min_distance = 0.2):
    """
    reduce truncated SVD components down to 3 dimensional space using umap algorithm.
    n_neighbours looks at how many neighbours to consider. A higher value implies capturing global patterns
    min_distance draws output points as being either packed or spread out.
    return umap as dataframe where columns are the coordinates of 2 dimenisonal space
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=neighbours,
        min_dist=min_distance,
        metric='cosine'
    )

    vectors_umap = reducer.fit_transform(vectors)

    return pd.DataFrame(vectors_umap, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'], index = vectors.index)
