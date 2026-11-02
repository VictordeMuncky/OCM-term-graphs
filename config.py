from pathlib import Path

DATA_FILE = Path("sample/fake_ethnotext.csv.gz")   
CODEBOOK = Path("ocmdefs.txt")
NROWS = None                                        # on ethnolab: 200_000 for the test run
CHUNK_SIZE = 100_000
DERIVED = Path("derived")
RESULTS = Path("results")

MIN_CODE_COUNT = 5     # a code must appear in at least this many paragraphs
MIN_COUNT = 2          # positive edges: pair must co-occur at least this often
MIN_NPMI = 0.3         # positive edges: minimum strength
MIN_EXPECTED = 5       # negative edges: expected co-occurrences needed before a low count means anything
MAX_NPMI = -0.3        # negative edges: maximum (most negative) strength
SAMPLE_FILTER = None   # e.g. "SCCS" to use only documents in that sample