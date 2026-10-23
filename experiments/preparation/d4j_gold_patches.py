"""Derive buggy->fixed gold patches from Defects4J's own patch files.

Defects4J ships framework/projects/<P>/patches/<bug>.src.patch as a fixed->buggy
diff. SE-Jury's bach_data.pkl "human patch" field flips the +/- lines of that
file but leaves the hunk headers unchanged, so `patch` rejects 66/100 of them as
malformed. This script reverses the Defects4J patch properly (headers, ranges
and line prefixes) and writes raw/SE-Jury/apr/d4j_gold_patches.json
({"<Project>-<bug>": diff}). The builder uses these as the reference shown to
the judge.

Usage (from experiments/):
  python preparation/d4j_gold_patches.py --patches /path/to/d4j-patches
"""

import argparse
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument(
    "--patches", required=True, help="dir with <Project>/<bug>.src.patch copied from the D4J image"
)
ap.add_argument("--out", default="raw/SE-Jury/apr/d4j_gold_patches.json")
a = ap.parse_args()


def reverse(diff):
    out = []
    for l in diff.split("\n"):
        if l.startswith("--- ") or l.startswith("+++ "):
            out.append(l)
            continue  # same path on both sides; keep a/ b/ labels
        m = re.match(r"@@ -(\S+) \+(\S+) @@(.*)", l)
        if m:
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
            continue
        if l.startswith("+"):
            out.append("-" + l[1:])
            continue
        if l.startswith("-"):
            out.append("+" + l[1:])
            continue
        out.append(l)
    return "\n".join(out)


res = {}
for f in sorted(glob.glob(os.path.join(a.patches, "*", "*.src.patch"))):
    proj = os.path.basename(os.path.dirname(f))
    bug = os.path.basename(f).split(".")[0]
    res[f"{proj}-{bug}"] = reverse(open(f, encoding="utf-8", errors="replace").read())
json.dump(res, open(a.out, "w"), indent=0)
print(len(res), "gold patches ->", a.out)
