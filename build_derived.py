import pandas as pd

from config import DATA_FILE, NROWS, CHUNK_SIZE, DERIVED

selected_columns = ['ocms', 'type', 'sreid', 'hdoc', 'culture', 'samples', 'mainowc', 'section', 'sectpar'] # other possibly important columns: important columns  'sreprev', 'srenext', 'parent', 'sectgpar', 'division','coverage'

chunk_list = []

for chunk in pd.read_csv(DATA_FILE, usecols=selected_columns, chunksize=CHUNK_SIZE, nrows=NROWS):
    chunk = chunk[(chunk['type'] == 'p')].copy()
    chunk['ocms'] = chunk['ocms'].fillna('').str.split().apply(lambda x: [i[1:] for i in x if i != '#000'])
    chunk_list.append(chunk)

codes_derived = pd.concat(chunk_list, ignore_index=True)
codes_derived = codes_derived[codes_derived['ocms'].str.len() > 0]

DERIVED.mkdir(exist_ok=True)
codes_derived.to_parquet(DERIVED / "codes.parquet", index=False)
print(f"saved {len(codes_derived):,} coded paragraphs to {DERIVED / 'codes.parquet'}")