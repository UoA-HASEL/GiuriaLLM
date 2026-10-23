"""Build the CoNaLa pointwise dataset from SE-Jury's per-annotator grades
(conala-human-grades.json; grades originally from Evtikhiev et al., "Out of the BLEU").

Setting follows SE-Jury's CoNaLa direct-assess task: the judge sees the intent
(problem), one generated snippet from one of five systems, and the reference
snippet, and grades Usefulness 0-4. The human reference snippet is never a
candidate.

Sampling: keep cells with >= --min-raters graders; agreement = 1 - mean|g - mean|/2
(the framework's numeric definition, span 4); stratum = high if agreement >=
--threshold else low. Draw --per-cell cells per (stratum, system) at random
(seed), using each intent at most once overall.

Usage (from experiments/):
  python preparation/build_conala_pointwise.py --per-cell 10 --threshold 0.8 --seed 0
Outputs: data/conala-pointwise.jsonl (id, prompt, code, reference),
data/conala-pointwise-human.jsonl (labels = all grader scores),
data/conala-pointwise-meta.csv.
"""

import argparse
import csv
import json
import random
from collections import Counter

SYSTEMS = ["baseline", "tranx-annot", "best-tranx", "best-tranx-rerank", "codex"]

ap = argparse.ArgumentParser()
ap.add_argument("--grades", default="raw/SE-Jury/conala/conala-human-grades.json")
ap.add_argument(
    "--aggregated",
    default="raw/SE-Jury/conala/conala_grade.json",
    help="SE-Jury M-MSR aggregated scores; recorded in meta only",
)
ap.add_argument("--per-cell", type=int, default=10, help="cells per (stratum, system)")
ap.add_argument("--threshold", type=float, default=0.8)
ap.add_argument("--min-raters", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-prefix", default="data/conala-pointwise")
a = ap.parse_args()
rng = random.Random(a.seed)

data = json.load(open(a.grades, encoding="utf-8"))
agg = json.load(open(a.aggregated, encoding="utf-8"))
assert [d["intent"] for d in data] == [d["intent"] for d in agg]


def agreement(v):
    m = sum(v) / len(v)
    return 1 - sum(abs(x - m) for x in v) / len(v) / 2.0


pool = {}  # (stratum, system) -> list of cells
for i, d in enumerate(data):
    for s in SYSTEMS:
        g = d.get("grade-" + s) or {}
        if len(g) < a.min_raters:
            continue
        ag = agreement(list(g.values()))
        st = "high" if ag >= a.threshold else "low"
        pool.setdefault((st, s), []).append(
            {"i": i, "system": s, "grades": g, "agreement": ag, "stratum": st}
        )

used, picked = set(), []
for st in ("high", "low"):
    for s in SYSTEMS:
        cells = pool[(st, s)]
        rng.shuffle(cells)
        n = 0
        for c in cells:
            if c["i"] in used:
                continue
            used.add(c["i"])
            picked.append(c)
            n += 1
            if n == a.per_cell:
                break
        assert n == a.per_cell, (st, s, n)

picked.sort(key=lambda c: (c["i"], SYSTEMS.index(c["system"])))
items, labels, meta = [], [], []
for c in picked:
    d = data[c["i"]]
    s = c["system"]
    iid = f"conala-{c['i']}-{s}"
    items.append({"id": iid, "prompt": d["intent"], "code": d[s], "reference": d["snippet"][0]})
    labels.append(
        {"id": iid, "labels": list(c["grades"].values()), "annotators": list(c["grades"])}
    )
    v = list(c["grades"].values())
    meta.append(
        {
            "id": iid,
            "intent_idx": c["i"],
            "system": s,
            "n_raters": len(v),
            "agreement": round(c["agreement"], 4),
            "stratum": c["stratum"],
            "mean_label": round(sum(v) / len(v), 4),
            "mmsr": agg[c["i"]]["grade-" + s],
            "graders": " ".join(c["grades"]),
        }
    )

with open(a.out_prefix + ".jsonl", "w", encoding="utf-8") as f:
    for r in items:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(a.out_prefix + "-human.jsonl", "w", encoding="utf-8") as f:
    for r in labels:
        f.write(json.dumps(r) + "\n")
with open(a.out_prefix + "-meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0]))
    w.writeheader()
    w.writerows(meta)

print(f"{len(items)} cells; intents used once: {len(used) == len(items)}")
print("by stratum x system:", sorted(Counter((m["stratum"], m["system"]) for m in meta).items()))
print("raters per cell:", sorted(Counter(m["n_raters"] for m in meta).items()))
print(
    "mean label by stratum:",
    {
        st: round(sum(m["mean_label"] for m in meta if m["stratum"] == st) / 50, 2)
        for st in ("high", "low")
    },
)
print("label value counts:", sorted(Counter(x for l in labels for x in l["labels"]).items()))
