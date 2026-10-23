"""Independent check of data/qure-listwise.* against raw/QuRE/QuRE.csv."""

import csv
import json
import re
from collections import Counter

raw = {r["id"]: r for r in csv.DictReader(open("raw/QuRE/QuRE.csv", encoding="utf-8"))}
items = [json.loads(l) for l in open("data/qure-listwise.jsonl")]
human = {
    json.loads(l)["id"]: json.loads(l)["labels"] for l in open("data/qure-listwise-human.jsonl")
}
meta = {r["id"]: r for r in csv.DictReader(open("data/qure-listwise-meta.csv"))}
errs = []
used_ids = Counter()
used_text = Counter()
words = Counter()
pos = Counter()
assert len(items) == 100 and len(human) == 100 and len(meta) == 100, (
    len(items),
    len(human),
    len(meta),
)
assert len({i["id"] for i in items}) == 100
for it in items:
    iid = it["id"]
    m = meta[iid]
    lab = human[iid]
    word = m["weak_word"]
    src = m["source_ids"].split()
    assert set(it) == {"id", "prompt", "r0", "r1", "r2", "r3", "r4"}, iid
    assert len(src) == 5 and len(lab) == 1 and int(m["defect_pos"]) == lab[0], iid
    assert re.fullmatch(rf"qure-{re.escape(word)}-\d+", iid), iid
    assert f'weak word "{word}"' in it["prompt"], iid
    ndef = 0
    for k, sid in enumerate(src):
        r = raw[sid]
        txt = " ".join(r["requirement"].split())
        if it[f"r{k}"] != txt:
            errs.append(f"{iid} r{k} text mismatch")
        if r["weak_word"] != word:
            errs.append(f"{iid} r{k} word {r['weak_word']}!={word}")
        if word[:3].lower() not in txt.lower() and word != "bad":
            errs.append(f"{iid} r{k} word not in text")
        isdef = r["defect"] == "defect"
        ndef += isdef
        if isdef != (k == lab[0]):
            errs.append(f"{iid} r{k} defect={isdef} label={lab[0]}")
        used_ids[sid] += 1
        used_text[txt] += 1
    if ndef != 1:
        errs.append(f"{iid} ndef={ndef}")
    if len({it[f"r{k}"] for k in range(5)}) != 5:
        errs.append(f"{iid} duplicate text in list")
    words[word] += 1
    pos[lab[0]] += 1
dup_ids = [k for k, v in used_ids.items() if v > 1]
dup_txt = [k for k, v in used_text.items() if v > 1]
print("errors:", errs[:10], len(errs))
print("reused source ids:", len(dup_ids), "reused texts:", len(dup_txt))
print("words:", len(words), sorted(words.items(), key=lambda x: (-x[1], x[0])))
print("defect pos:", sorted(pos.items()))
print("max len:", max(len(it[f"r{k}"]) for it in items for k in range(5)))
# overall raw defect rate vs per-list 1/5 - just for context
raw_rows = list(raw.values())
print("raw rows", len(raw_rows), "defect", sum(r["defect"] == "defect" for r in raw_rows))
