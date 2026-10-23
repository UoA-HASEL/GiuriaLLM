"""Execution oracle for data/apr-pointwise.jsonl inside the Defects4J container.

For each item: check out the buggy version, run the trigger tests (expect
failures); apply the human gold patch (expect the trigger tests to pass); apply
the tool patch (expect it to apply, compile and pass the trigger tests).
Results go to --out (JSONL), one line per item; nothing is written to the data.

Run inside the container (see scripts/run_apr_oracle.sh):
  python3 scripts/verify_apr_d4j.py --work /work --out /work/apr-oracle.jsonl [--ids apr-1 apr-5]
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/apr-pointwise.jsonl")
ap.add_argument("--meta", default="data/apr-pointwise-meta.csv")
ap.add_argument("--work", default="/work")
ap.add_argument("--out", default="/work/apr-oracle.jsonl")
ap.add_argument("--ids", nargs="*")
ap.add_argument("--timeout", type=int, default=1800)
a = ap.parse_args()


def sh(cmd, cwd=None, timeout=None):
    r = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout or a.timeout
    )
    return r.returncode, r.stdout + r.stderr


def trigger_tests(ctx):
    # lines like " - org.apache.commons.math.util.MathUtilsTest::testGcd"
    return re.findall(r"^\s*-\s+([\w.$]+::\w+)", ctx, re.M)


def run_tests(d, tests):
    """Return (n_failing, n_run_ok, log). Uses defects4j test -t per trigger test."""
    fails = 0
    log = ""
    for t in tests:
        rc, out = sh(f"defects4j test -t {t}", cwd=d)
        m = re.search(r"Failing tests:\s*(\d+)", out)
        n = int(m.group(1)) if m else None
        log += f"[{t}] rc={rc} failing={n}\n"
        if n is None or n > 0:
            fails += 1
    return fails, log


def _normalise_crlf(d, diff, strip):
    # Some Defects4J sources use CRLF while the diffs are LF: normalise the touched files.
    for m in re.finditer(r"^\+\+\+ (\S+)", diff, re.M):
        fp = os.path.join(d, *m.group(1).split("/")[strip:])
        if os.path.isfile(fp):
            b = open(fp, "rb").read()
            if b"\r\n" in b:
                open(fp, "wb").write(b.replace(b"\r\n", b"\n"))


def apply_patch(d, diff, strip=None):
    """strip=None: try -p0 (svn-style human patches) then -p1 (git-style)."""
    pf = os.path.join(d, "_oracle.diff")
    open(pf, "w").write(diff + "\n")
    tried = []
    # Defects4J ships its human patches as fixed->buggy diffs, so also try reversed (-R).
    for st in [strip] if strip is not None else [0, 1]:
        _normalise_crlf(d, diff, st)
        for rev in ("", " -R"):
            rc, out = sh(
                f"patch -p{st}{rev} -l -N --no-backup-if-mismatch --dry-run < _oracle.diff", cwd=d
            )
            tried.append(f"-p{st}{rev}: " + out[-400:])
            if rc == 0:
                rc, out = sh(
                    f"patch -p{st}{rev} -l -N --no-backup-if-mismatch < _oracle.diff", cwd=d
                )
                return rc == 0, f"applied with -p{st}{rev}"
    return False, "\n".join(tried)


items = {json.loads(l)["id"]: json.loads(l) for l in open(a.data, encoding="utf-8")}
meta = {m["id"]: m for m in csv.DictReader(open(a.meta, encoding="utf-8"))}
ids = a.ids or list(items)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
os.makedirs(a.work, exist_ok=True)

for iid in ids:
    if iid in done:
        continue
    it, m = items[iid], meta[iid]
    proj, bug = m["project"], m["bug_id"]
    t0 = time.time()
    res = {"id": iid, "project": proj, "bug": bug}
    tests = trigger_tests(it["prompt"])
    res["n_trigger"] = len(tests)
    base = os.path.join(a.work, iid)  # per item: several tool patches can target one bug
    try:
        if not os.path.isdir(base):
            rc, out = sh(f"defects4j checkout -p {proj} -v {bug}b -w {base}")
            if rc != 0:
                res["error"] = "checkout: " + out[-500:]
                raise RuntimeError
            rc, out = sh("defects4j compile", cwd=base)
            if rc != 0:
                res["error"] = "compile buggy: " + out[-500:]
                raise RuntimeError
        # 1. buggy version fails the trigger tests
        nf, log = run_tests(base, tests)
        res["buggy_failing"] = nf
        # 2. gold patch
        gd = base + "_gold"
        shutil.rmtree(gd, ignore_errors=True)
        shutil.copytree(base, gd, symlinks=True)
        ok, out = apply_patch(gd, it["reference"], None)
        res["gold_applies"] = ok
        res["gold_apply_log"] = out
        if ok:
            rc, out = sh("defects4j compile", cwd=gd)
            res["gold_compiles"] = rc == 0
            if rc == 0:
                nf, log = run_tests(gd, tests)
                res["gold_failing"] = nf
        shutil.rmtree(gd, ignore_errors=True)
        # 3. tool patch (diff -w -r -u <proj><bug>b/... <proj><bug>b_PatchN/...)
        td = base + "_tool"
        shutil.rmtree(td, ignore_errors=True)
        shutil.copytree(base, td, symlinks=True)
        ok, out = apply_patch(td, it["patch"], 1)
        res["tool_applies"] = ok
        res["tool_apply_mode"] = out if ok else None
        if not ok:
            res["tool_apply_log"] = out
        else:
            rc, out = sh("defects4j compile", cwd=td)
            res["tool_compiles"] = rc == 0
            if rc == 0:
                nf, log = run_tests(td, tests)
                res["tool_failing"] = nf
            else:
                res["tool_compile_log"] = out[-500:]
        shutil.rmtree(td, ignore_errors=True)
    except RuntimeError:
        pass
    except subprocess.TimeoutExpired:
        res["error"] = "timeout"
    shutil.rmtree(base, ignore_errors=True)
    res["seconds"] = round(time.time() - t0)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res)[:300], flush=True)
