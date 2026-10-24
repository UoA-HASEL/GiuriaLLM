#!/usr/bin/env bash
# Exercise the published CLI, data mounts, and outputs without API calls.
set -euo pipefail
image=${1:-giuriallm:local}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/runs"
chmod 755 "$work"
run=(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$work:/work" "$image")
docker run --rm --network none "$image" --help
docker run --rm --network none --entrypoint python "$image" \
  -c 'import anthropic, openai, httpx; from importlib.metadata import version; print(version("giuriallm"))'
docker run --rm --network none "$image" validate \
  /opt/giuriallm/demo/configs/apr-patch.yaml --check-dataset
# Check the documented copy-out command, including the bundled data paths.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$work:/work" --entrypoint cp "$image" -R /opt/giuriallm/demo /work/demo
"${run[@]}" validate demo/configs/apr-patch.yaml --check-dataset
# Run both bundled APR configurations with mock judges, without network access.
for config in apr-patch.yaml apr-five-judges.yaml; do
  directory=$("${run[@]}" run "/opt/giuriallm/demo/configs/$config" --provider mock --no-cache --quiet)
  "${run[@]}" show "$directory"
done
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$work:/work" --entrypoint python "$image" -c '
import json
from pathlib import Path
directories = list(Path("runs").iterdir())
assert len(directories) == 2
counts = []
for directory in directories:
    for name in ("config.yaml", "manifest.json", "judgements.jsonl", "results.json", "report.md", "console.txt"):
        assert (directory / name).stat().st_size > 0, (directory, name)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["counts"]["items"] == 1, manifest["counts"]
    assert manifest["counts"]["failed_judgements"] == 0, manifest["counts"]
    counts.append(manifest["counts"]["judgements"])
assert sorted(counts) == [1, 5]
print("All example configurations produced valid results and persisted reports.")
'
