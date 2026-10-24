"""Build the QuRE listwise dataset (Femmer et al., arXiv 2508.08868).

Each item is a list of K requirements that share one weak word: exactly one is
labelled `defect`, the other K-1 are `ok`. The judge ranks the K requirements by
how likely the weak word is a defect in context; meta-evaluation scores the top-1
against the index of the defective requirement.

Construction: dedupe by requirement text; group by weak word; keep words with >=1
defect and >=K-1 ok; build up to --per-word lists per word (each requirement used
once); then trim from the words with the most lists down to --total.

Usage (from experiments/):
  python preparation/build_qure_listwise.py --csv raw/QuRE/QuRE.csv --per-word 6 --total 100 \
      --list-size 5 --seed 0 --out-prefix data/qure-listwise
Outputs: <prefix>.jsonl, <prefix>-human.jsonl (labels = index of the defect),
<prefix>-meta.csv (weak_word, defect position, source ids).
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="raw/QuRE/QuRE.csv")
ap.add_argument("--per-word", type=int, default=6, help="max lists per weak word")
ap.add_argument("--total", type=int, default=100, help="trim to this many lists (0 = no trim)")
ap.add_argument("--list-size", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-prefix", default="data/qure-listwise")
a = ap.parse_args()
rng = random.Random(a.seed)
K = a.list_size

seen, rows = set(), []
for r in csv.DictReader(open(a.csv, encoding="utf-8")):
    t = " ".join(r["requirement"].split())
    if t in seen:
        continue
    seen.add(t)
    rows.append(
        {"id": r["id"], "text": t, "defect": r["defect"] == "defect", "word": r["weak_word"]}
    )

by_word = defaultdict(lambda: {"defect": [], "ok": []})
for r in rows:
    by_word[r["word"]]["defect" if r["defect"] else "ok"].append(r)

lists = defaultdict(list)  # word -> list of candidate lists (defect first)
for word in sorted(by_word):
    d, o = by_word[word]["defect"], by_word[word]["ok"]
    rng.shuffle(d)
    rng.shuffle(o)
    n = min(a.per_word, len(d), len(o) // (K - 1))
    if n == 0:
        print(f"skip {word}: {len(d)} defect / {len(o)} ok")
        continue
    for k in range(n):
        lists[word].append([d[k]] + o[k * (K - 1) : (k + 1) * (K - 1)])

if a.total:
    while sum(len(v) for v in lists.values()) > a.total:
        word = max(
            sorted(lists), key=lambda w: len(lists[w])
        )  # most lists; ties -> alphabetical first
        lists[word].pop()

items, labels, meta = [], [], []
for word in sorted(lists):
    for k, pool in enumerate(lists[word]):
        pos = rng.randrange(K)
        cands = pool[1:]
        cands.insert(pos, pool[0])
        item_id = f"qure-{word}-{k}"
        row = {
            "id": item_id,
            "prompt": (
                f"All {K} requirements below come from automotive specifications and "
                f'contain a form of the weak word "{word}" (inflected or derived forms count, '
                f"e.g. comparatives, superlatives and adverbs such as longer, smallest, worst). "
                f"Exactly one of them was flagged by "
                f"industrial reviewers because the weak word makes the requirement "
                f"ambiguous or untestable in its context; in the others the word is "
                f"used in a fixed term or is otherwise precise enough."
            ),
        }
        for i, c in enumerate(cands):
            row[f"r{i}"] = c["text"]
        items.append(row)
        labels.append({"id": item_id, "labels": [pos]})
        meta.append(
            {
                "id": item_id,
                "weak_word": word,
                "defect_pos": pos,
                "source_ids": " ".join(c["id"] for c in cands),
            }
        )

with open(f"{a.out_prefix}.jsonl", "w") as f:
    for r in items:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{a.out_prefix}-human.jsonl", "w") as f:
    for r in labels:
        f.write(json.dumps(r) + "\n")
with open(f"{a.out_prefix}-meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "weak_word", "defect_pos", "source_ids"])
    w.writeheader()
    w.writerows(meta)
print(f"{len(items)} lists, {len(lists)} weak words, seed={a.seed}, per-word cap={a.per_word}")
print(
    "lists per word:",
    dict(sorted(Counter(m["weak_word"] for m in meta).items(), key=lambda kv: -kv[1])),
)
print("defect position counts:", dict(sorted(Counter(m["defect_pos"] for m in meta).items())))
print("max requirement chars:", max(len(r[f"r{i}"]) for r in items for i in range(K)))
