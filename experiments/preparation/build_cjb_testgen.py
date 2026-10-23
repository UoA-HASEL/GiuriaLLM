"""Build the CodeJudgeBench TestGen pairwise dataset (Jiang et al., arXiv 2507.10535).

Each item: a competitive-programming problem statement and two candidate unit
tests (input + expected output). One expected output is correct, the other
wrong; the two tests use different inputs (CJB Sec. 3.4). The judge picks the
correct test; no reference code is given.

Sampling: only the Gemini splits (gemini_2.5_flash, gemini_2.5_pro), so the
configured judges do not evaluate their own outputs. Within each split sort by contest_date (newest
first); take questions alternately from the splits, each question_id at most
once, until --per-split per split. The correct test is placed in position A or B
balanced within each split (seed).

Usage (from experiments/):
  python preparation/build_cjb_testgen.py --per-split 50 --seed 0
Outputs: data/cjb-testgen.jsonl (id, prompt, test_a, test_b),
data/cjb-testgen-human.jsonl (labels = index of the correct test),
data/cjb-testgen-meta.csv.
"""

import argparse
import csv
import json
import random
from collections import Counter

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="raw/CodeJudgeBench/testgen")
ap.add_argument("--splits", nargs="+", default=["gemini_2.5_flash", "gemini_2.5_pro"])
ap.add_argument("--per-split", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-prefix", default="data/cjb-testgen")
ap.add_argument(
    "--exclude",
    nargs="*",
    default=["abc398_d"],
    help="question_ids dropped after the LiveCodeBench oracle check (scripts/verify_cjb_testgen.py): "
    "abc398_d's 'correct' test states expected output '0' where the true output is a 20-character "
    "string of zeros (CJB's lenient numeric comparison accepted it)",
)
a = ap.parse_args()
rng = random.Random(a.seed)

queues = {}
for s in a.splits:
    df = pd.read_parquet(f"{a.dir}/{s}-00000-of-00001.parquet")
    df["contest_date"] = pd.to_datetime(df["contest_date"])
    df = df.sort_values(["contest_date", "question_id"], ascending=[False, True])
    queues[s] = df.to_dict("records")

used, picked = set(a.exclude), {s: [] for s in a.splits}
while any(len(picked[s]) < a.per_split for s in a.splits):
    progressed = False
    for s in a.splits:
        if len(picked[s]) >= a.per_split:
            continue
        while queues[s]:
            r = queues[s].pop(0)
            if r["question_id"] not in used:
                used.add(r["question_id"])
                picked[s].append(r)
                progressed = True
                break
    if not progressed:
        raise SystemExit("ran out of questions: {s: len(v) for s, v in picked.items()}")


def fmt_test(inp, out):
    return f"Input:\n{inp.rstrip()}\n\nExpected output:\n{out.rstrip()}"


items, labels, meta = [], [], []
for s in a.splits:
    pos = [0] * (a.per_split // 2) + [1] * (a.per_split - a.per_split // 2)
    rng.shuffle(pos)
    for r, p in zip(picked[s], pos):
        iid = f"cjb-{s}-{r['question_id']}"
        ctx = f"# {r['question_title']}\n\n{r['question_content'].strip()}"
        if r["starter_code"].strip():
            ctx += f"\n\nStarter code:\n{r['starter_code'].strip()}"
        good = fmt_test(r["pos_input"], r["pos_response"])
        bad = fmt_test(r["neg_input"], r["neg_response"])
        items.append(
            {
                "id": iid,
                "prompt": ctx,
                "test_a": good if p == 0 else bad,
                "test_b": bad if p == 0 else good,
            }
        )
        labels.append({"id": iid, "labels": [p]})
        meta.append(
            {
                "id": iid,
                "split": s,
                "question_id": r["question_id"],
                "platform": r["platform"],
                "difficulty": r["difficulty"],
                "contest_date": r["contest_date"].date().isoformat(),
                "correct_pos": p,
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

print(f"{len(items)} items; unique questions: {len(used) - len(a.exclude)}; excluded: {a.exclude}")
for s in a.splits:
    ms = [m for m in meta if m["split"] == s]
    print(
        f"  {s}: {len(ms)}, dates {min(m['contest_date'] for m in ms)}..{max(m['contest_date'] for m in ms)},",
        "difficulty",
        dict(Counter(m["difficulty"] for m in ms)),
        "platform",
        dict(Counter(m["platform"] for m in ms)),
        "correct_pos",
        dict(Counter(m["correct_pos"] for m in ms)),
    )
print("max prompt chars:", max(len(i["prompt"]) for i in items))
