"""Oracle check for data/cjb-testgen.jsonl against LiveCodeBench test cases.

For every item, look up the LCB problem (public + private tests), and check:
  - the correct test's input exists in LCB and its expected output equals LCB's;
  - the wrong test's input exists in LCB and its expected output differs from LCB's.
Usage (from experiments/): python scripts/verify_cjb_testgen.py --lcb DIR
"""

import argparse
import base64
import csv
import json
import pickle
import zlib

ap = argparse.ArgumentParser()
ap.add_argument(
    "--lcb", required=True, help="Directory containing trusted LiveCodeBench test files"
)
ap.add_argument("--files", nargs="+", default=["test5.jsonl", "test6.jsonl"])
a = ap.parse_args()


def norm(s):
    return "\n".join(l.rstrip() for l in s.strip().splitlines())


def loose(s):
    try:  # functional (LeetCode) outputs are JSON values; compare parsed
        parsed = json.loads(s)
    except Exception:
        pass
    else:
        # LCB serializes strings with quotes; CJB sometimes stores their raw text.
        if isinstance(parsed, str):
            s = parsed
        else:
            return json.dumps(parsed, sort_keys=True)
    # CJB compares generated outputs to ground truth leniently; mirror that:
    # whitespace-insensitive token sequence, case-insensitive.
    return " ".join(s.split()).lower()


def load_tests(d):
    tests = json.loads(d["public_test_cases"])
    priv = d["private_test_cases"]
    try:
        tests += json.loads(priv)
    except Exception:
        tests += json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode("utf-8")))))
    return {norm(t["input"]): norm(t["output"]) for t in tests}


lcb = {}
for f in a.files:
    for l in open(f"{a.lcb}/{f}", encoding="utf-8"):
        d = json.loads(l)
        lcb[d["question_id"]] = d
items = {
    json.loads(l)["id"]: json.loads(l) for l in open("data/cjb-testgen.jsonl", encoding="utf-8")
}
human = {
    json.loads(l)["id"]: json.loads(l)["labels"][0]
    for l in open("data/cjb-testgen-human.jsonl", encoding="utf-8")
}
meta = list(csv.DictReader(open("data/cjb-testgen-meta.csv", encoding="utf-8")))


def split(test):
    i = test.index("\n\nExpected output:\n")
    return norm(test[len("Input:\n") : i]), norm(test[i + len("\n\nExpected output:\n") :])


stats = {
    "ok": 0,
    "no_problem": [],
    "input_missing": [],
    "correct_mismatch": [],
    "wrong_matches": [],
    "n_tests": [],
}
for m in meta:
    iid, qid = m["id"], m["question_id"]
    if qid not in lcb:
        stats["no_problem"].append(iid)
        continue
    tests = load_tests(lcb[qid])
    stats["n_tests"].append(len(tests))
    it = items[iid]
    good_key = "test_a" if human[iid] == 0 else "test_b"
    bad_key = "test_b" if good_key == "test_a" else "test_a"
    gi, go = split(it[good_key])
    bi, bo = split(it[bad_key])
    bad = False
    if gi not in tests or bi not in tests:
        stats["input_missing"].append(iid)
        bad = True
    else:
        if loose(tests[gi]) != loose(go):
            stats["correct_mismatch"].append((iid, go[:60], tests[gi][:60]))
            bad = True
        if loose(tests[bi]) == loose(bo):
            stats["wrong_matches"].append((iid, bo[:40]))
            bad = True
    if not bad:
        stats["ok"] += 1
print("comparison: whitespace- and case-insensitive token match")
print("items fully verified:", stats["ok"], "/", len(meta))
print(
    "tests per problem min/median:",
    min(stats["n_tests"]),
    sorted(stats["n_tests"])[len(stats["n_tests"]) // 2],
)
for k in ("no_problem", "input_missing", "correct_mismatch", "wrong_matches"):
    print(f"{k}: {len(stats[k])}", stats[k][:10])
