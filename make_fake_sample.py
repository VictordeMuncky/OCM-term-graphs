"""Build a fake ethnotext-style CSV so code can be developed without real HRAF data.

Same 30 columns as ethnotext_v430.csv.gz, same "#224 #462" code format, invented text.
Known answer for testing: the codes in PLANTED appear ONLY together, in exactly
PLANTED_N paragraphs, so a correct co-occurrence count must report exactly that.
"""
import random
import re
import uuid
from pathlib import Path

import pandas as pd

COLUMNS = ["text", "ocms", "type", "pageEid", "prevPage", "nextPage", "uuid", "sreid",
           "sreprev", "srenext", "parent", "section", "sectpar", "sectgpar", "division",
           "culture", "coverage", "place", "page", "roll", "hdoc", "title", "byline",
           "pub.date", "pub.lang", "field.date", "pub.type", "owcs", "mainowc", "samples"]
PLANTED = ("533", "780")   # MUSIC and RELIGIOUS PRACTICES
PLANTED_N = 7


def load_codes(path):
    """All OCM code numbers from the codebook."""
    with open(path, encoding="utf-8") as f:
        return [m.group(1) for block in f.read().split("\n\n")
                if (m := re.match(r"(\d{3,4})\s", block.strip()))]


def make_fake(codebook="ocmdefs.txt", out="sample/fake_ethnotext.csv.gz", n_docs=6, seed=0):
    rng = random.Random(seed)                       # fixed seed = the same file every time
    codes = [c for c in load_codes(codebook) if c not in ("000", *PLANTED)]
    rows = []

    for d in range(n_docs):
        hdoc = f"fk{d:02d}-001"
        doc = []
        for g in range(rng.randint(2, 3)):                  # top-level sections
            for p in range(rng.randint(1, 3)):              # sub-sections
                for s in range(rng.randint(1, 3)):          # sections
                    sec = f"{hdoc}-g{g}-p{p}-s{s}"
                    theme = rng.sample(codes, 5)            # codes that recur in this section
                    for _ in range(rng.randint(2, 6)):      # paragraphs
                        chosen = set(rng.sample(theme, rng.randint(1, 3)))
                        if rng.random() < 0.15:
                            chosen.add(rng.choice(codes))   # occasional stray code
                        if rng.random() < 0.03:
                            chosen = set()                  # occasional uncoded paragraph
                        doc.append(dict(type="p", section=sec, sectpar=f"{hdoc}-g{g}-p{p}",
                                        sectgpar=f"{hdoc}-g{g}", codes=chosen))
        for _ in range(rng.randint(2, 4)):                  # bibliography at the end
            doc.append(dict(type="bibl.item", section=f"{hdoc}-bib", sectpar=None,
                            sectgpar=None, codes={"000"}))

        ids = [f"{hdoc}-sre{i:04d}" for i in range(len(doc))]
        for i, r in enumerate(doc):                         # link each SRE to its neighbours
            r.update(sreid=ids[i],
                     sreprev=ids[i - 1] if i > 0 else None,
                     srenext=ids[i + 1] if i < len(doc) - 1 else None,
                     hdoc=hdoc, culture=f"Fakeculture {d}", owcs=f"FK{d:02d}",
                     mainowc=f"FK{d:02d}", samples="SCCS" if d % 3 == 0 else None)
        rows.extend(doc)

    # The known answer: PLANTED codes go into exactly PLANTED_N paragraphs, and nowhere else
    paragraphs = [r for r in rows if r["type"] == "p"]
    for r in rng.sample(paragraphs, PLANTED_N):
        r["codes"] |= set(PLANTED)

    for i, r in enumerate(rows):                            # fill the remaining columns
        r["ocms"] = " ".join(f"#{c}" for c in sorted(r.pop("codes"), key=int)) or None
        r.update(text=f"Invented paragraph {i}, about nothing real.", uuid=str(uuid.uuid4()),
                 parent=r["section"], division="body", coverage="1950-1960", place="Nowhere",
                 page=i // 4 + 1, pageEid=f"pg{i // 4 + 1}", prevPage=None, nextPage=None,
                 roll=None, title=f"A Fake Ethnography of {r['culture']}", byline="Doe, J.",
                 **{"pub.date": "1970", "pub.lang": "eng", "field.date": "1965",
                    "pub.type": "Book"})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out, index=False)                             # .gz ending -> gzip-compressed
    print(f"wrote {len(df)} rows from {n_docs} documents to {out}")
    print(f"known answer: {PLANTED} co-occur in exactly {PLANTED_N} paragraphs, nowhere else")
    return df


if __name__ == "__main__":
    make_fake()