#!/usr/bin/env bash
# Run the four panel configs once and record the run ids.
#   usage: OPENAI_API_KEY=... AIHUBMIX_API_KEY=... scripts/run_panel.sh REP_LABEL [TASK...]
# Writes runs/panel-<REP_LABEL>.log and runs/panel-<REP_LABEL>.ids (task<TAB>run_dir).
# With TASK arguments only those tasks run and their lines in the .ids file are replaced.
set -euo pipefail
cd "$(dirname "$0")/.."
rep="${1:?rep label, e.g. rep2}"
shift
[[ "$rep" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid repetition label: $rep" >&2; exit 1; }
tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
  tasks=(conala-pointwise apr-pointwise qure-listwise cjb-testgen)
fi
for t in "${tasks[@]}"; do
  case "$t" in
    conala-pointwise|apr-pointwise|qure-listwise|cjb-testgen) ;;
    *) echo "Unknown task: $t" >&2; exit 1 ;;
  esac
  test -f "configs/$t-panel.yaml"
done
: "${OPENAI_API_KEY:?}" "${AIHUBMIX_API_KEY:?}"
mkdir -p runs
log="runs/panel-$rep.log"
ids="runs/panel-$rep.ids"
touch "$ids"
pending=$(mktemp "runs/.panel-$rep.XXXXXX")
trap 'rm -f "$pending"' EXIT
for t in "${tasks[@]}"; do
  echo "=== $t $(date -u +%FT%TZ)" | tee -a "$log"
  # --quiet provides the run directory without parsing the formatted report.
  d=$(giuriallm run "configs/$t-panel.yaml" --quiet 2> >(tee -a "$log" >&2))
  test -f "$d/manifest.json"
  printf '%s\n' "$d" | tee -a "$log"
  # Preserve the previous mapping until a replacement run has been saved.
  awk -F '\t' -v task="$t" '$1 != task' "$ids" > "$pending"
  printf '%s\t%s\n' "$t" "$d" >> "$pending"
  mv "$pending" "$ids"
done
echo "=== ALL DONE $(date -u +%FT%TZ)" | tee -a "$log"
