"""Build the APR pointwise dataset from SE-Jury's APR-Assess data
(bach_data.pkl + bach_label.csv; Defects4J patches with 41 annotators).

Setting follows SE-Jury's APR direct-assess task: the judge sees the failing
tests (root-cause note + failing test methods), the tool patch, and the human
gold patch (taken from Defects4J's own patch files, reversed to buggy->fixed,
because the pickle's human_patch text has malformed hunk headers), and decides whether the patch correctly fixes the bug. Human labels
are Y/N per annotator; '0' is mapped to N as in SE-Jury's eval scripts.

Pool: patches with patch_id < 1000 (Defects4J, failing tests available) whose
failing-test context is <= --max-ctx characters. Stratify by
(unanimous / dissent) x (majority Y / N); allocate --total proportionally
(largest remainder) and sample within cells (seed).

Usage (from experiments/):
  python preparation/build_apr_pointwise.py --total 100 --max-ctx 5000 --seed 0
Outputs: data/apr-pointwise.jsonl (id, prompt, patch, reference),
data/apr-pointwise-human.jsonl (labels = per-annotator 1/0),
data/apr-pointwise-meta.csv.
"""

import argparse
import csv
import json
import pickle
import random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", default="raw/SE-Jury/apr/bach_data.pkl")
ap.add_argument("--labels", default="raw/SE-Jury/apr/bach_label.csv")
ap.add_argument("--total", type=int, default=100)
ap.add_argument("--max-ctx", type=int, default=5000, help="max chars of failing-test context")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument(
    "--gold",
    default="raw/SE-Jury/apr/d4j_gold_patches.json",
    help="buggy->fixed gold patches derived from Defects4J (preparation/d4j_gold_patches.py); "
    "the pickle's human_patch field has malformed hunk headers",
)
ap.add_argument("--out-prefix", default="data/apr-pointwise")
ap.add_argument(
    "--exclude",
    nargs="*",
    default=[],
    help="patch ids dropped after the Defects4J execution oracle (scripts/verify_apr_d4j.py); "
    "the cell quota is refilled from the remaining pool",
)
a = ap.parse_args()
rng = random.Random(a.seed)

(
    pid,
    proj,
    bug,
    failed,
    failed_names,
    failed_content,
    _tests,
    human_patch,
    author_label,
    tool_patch,
) = pickle.load(open(a.pkl, "rb"))
gold = json.load(open(a.gold, encoding="utf-8"))
votes = defaultdict(list)
for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    votes[int(r["Patch"])].append((r["Annotator"], 1 if r["Answer"] == "Y" else 0))


def context(i):
    return (
        f"Project: {proj[i]}, bug {bug[i]} (Defects4J).\n\n"
        f"Failing tests before the patch:\n{''.join(failed[i]).strip()}\n\n"
        f"Failing test methods:\n" + "\n\n".join(t.strip() for t in failed_content[i])
    )


pool = defaultdict(list)
dropped = Counter()
for i in range(len(pid)):
    p = int(pid[i])
    if p >= 1000:
        dropped["no_failing_tests"] += 1
        continue
    if str(p) in a.exclude:
        dropped["excluded_by_oracle"] += 1
        continue
    if "@@" not in tool_patch[i]:  # 4+ pickle entries hold UTF-16 garbage instead of a diff
        dropped["patch_not_a_diff"] += 1
        continue
    ctx = context(i)
    if len(ctx) > a.max_ctx:
        dropped["ctx_too_long"] += 1
        continue
    v = [x for _, x in votes[p]]
    maj = "Y" if v.count(1) >= len(v) * 0.5 else "N"  # SE-Jury rule
    unan = "unanimous" if len(set(v)) == 1 else "dissent"
    pool[(unan, maj)].append((i, ctx, v, maj, unan))

cells = sorted(pool)
sizes = {c: len(pool[c]) for c in cells}
n_pool = sum(sizes.values())
quota = {c: a.total * sizes[c] // n_pool for c in cells}
rest = a.total - sum(quota.values())
for c in sorted(cells, key=lambda c: -(a.total * sizes[c] / n_pool - quota[c]))[:rest]:
    quota[c] += 1

picked = []
for c in cells:
    rows = pool[c][:]
    rng.shuffle(rows)
    picked.extend(rows[: quota[c]])
picked.sort(key=lambda r: int(pid[r[0]]))

items, labels, meta = [], [], []
for i, ctx, v, maj, unan in picked:
    iid = f"apr-{int(pid[i])}"
    items.append(
        {
            "id": iid,
            "prompt": ctx,
            "patch": tool_patch[i].strip("\n"),
            "reference": gold[f"{proj[i]}-{bug[i]}"].strip("\n"),
        }
    )
    labels.append({"id": iid, "labels": v, "annotators": [ann for ann, _ in votes[int(pid[i])]]})
    meta.append(
        {
            "id": iid,
            "project": proj[i],
            "bug_id": bug[i],
            "n_raters": len(v),
            "y_votes": sum(v),
            "majority": maj,
            "stratum": unan,
            "author_label": author_label[i].strip() or "n/a",
            "ctx_chars": len(ctx),
            "annotators": " ".join(ann for ann, _ in votes[int(pid[i])]),
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

print("dropped:", dict(dropped), "pool:", n_pool, {f"{u}/{m}": n for (u, m), n in sizes.items()})
print("quota:", {f"{u}/{m}": n for (u, m), n in quota.items()}, "total", len(items))
print("projects:", dict(Counter(m["project"] for m in meta)))
print("raters per patch:", sorted(Counter(m["n_raters"] for m in meta).items()))
print("author label vs majority:", dict(Counter((m["author_label"], m["majority"]) for m in meta)))
print(
    "max ctx / patch / reference chars:",
    max(m["ctx_chars"] for m in meta),
    max(len(i["patch"]) for i in items),
    max(len(i["reference"]) for i in items),
)
