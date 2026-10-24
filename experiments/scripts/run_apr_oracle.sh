#!/usr/bin/env bash
# Run the APR execution oracle in N parallel Defects4J containers.
# Usage: scripts/run_apr_oracle.sh [N=8] [OUTDIR=runs/apr-oracle]
set -euo pipefail
N=${1:-8}
VAL=$(cd "$(dirname "$0")/.." && pwd)
cd "$VAL"
OUT=${2:-runs/apr-oracle}
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "N must be a positive integer" >&2; exit 1; }
mkdir -p "$OUT/work"
OUT=$(cd "$OUT" && pwd)
python3 - "$N" "$OUT" <<'PY'
import json, sys
n, out = int(sys.argv[1]), sys.argv[2]
ids = [json.loads(l)["id"] for l in open("data/apr-pointwise.jsonl")]
for k in range(n):
    open(f"{out}/shard{k}.ids", "w").write(" ".join(ids[k::n]))
PY
for k in $(seq 0 $((N-1))); do
  docker run -d --rm --name apr-oracle-$k --network=host \
    -v "$VAL":/val:ro -v "$OUT":/out \
    giuriallm-d4j bash -lc "cd /val && python3 scripts/verify_apr_d4j.py --work /out/work/$k --out /out/shard$k.jsonl --ids \$(cat /out/shard$k.ids) > /out/shard$k.log 2>&1"
done
echo "started $N containers; logs in $OUT/shard*.log; merge with: cat $OUT/shard*.jsonl"
