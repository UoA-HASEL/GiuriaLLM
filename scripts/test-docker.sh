#!/usr/bin/env bash
# Exercise the published CLI, data mounts, and outputs without API calls.
set -euo pipefail
image=${1:-giuriallm:local}
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$root/examples" "$work/examples"
mkdir "$work/runs"
chmod 755 "$work"
run=(docker run --rm --network none --user "$(id -u):$(id -g)" -v "$work:/work" "$image")
docker run --rm --network none "$image" --help
docker run --rm --network none --entrypoint python "$image" \
  -c 'import anthropic, openai, httpx; from importlib.metadata import version; print(version("giuriallm"))'
docker run --rm --network none "$image" validate \
  /opt/giuriallm/examples/code-gen-pairwise.yaml --check-dataset
docker run --rm --network none "$image" validate \
  /opt/giuriallm/demo/configs/apr-patch.yaml --check-dataset
# Check the documented copy-out command, including the bundled data paths.
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$work:/work" --entrypoint cp "$image" -R /opt/giuriallm/demo /work/demo
"${run[@]}" validate demo/configs/apr-patch.yaml --check-dataset
for config in "$work"/examples/*.yaml; do
  "${run[@]}" validate "examples/$(basename "$config")" --check-dataset
  directory=$("${run[@]}" run "examples/$(basename "$config")" --provider mock --no-cache --quiet)
  "${run[@]}" show "$directory"
done
# A test-only provider exercises the real APR input and original engine offline.
# It is injected in this test process only; the shipped CLI still calls the models.
docker run --rm -i --network none --user "$(id -u):$(id -g)" \
  -v "$work:/work" --entrypoint python "$image" - <<'PY'
import asyncio
from pathlib import Path
from giuriallm.engine import Engine, write_run
from giuriallm.providers import Provider, LLMResponse
from giuriallm.report import write_reports
from giuriallm.spec import load_config

class TestProvider(Provider):
    name = "openai"
    def __init__(self):
        self.models = []
    async def complete(self, request):
        self.models.append(request.model.id)
        return LLMResponse(payload={"scores": [{"criterion": "functional_correctness", "score": 1}]})

for filename, count in (("apr-patch.yaml", 1), ("apr-five-judges.yaml", 5)):
    config = load_config(Path("/opt/giuriallm/demo/configs") / filename)
    provider = TestProvider()
    result = asyncio.run(Engine(config, provider=provider).run())
    assert result.n_items == 1 and result.n_judgements == count and result.n_failed == 0
    assert sorted(provider.models) == sorted(model.id for model in config.judges())
    assert result.results[0].aggregate.score_for("functional_correctness") == 1
    directory = write_run(config, result, Path("runs"))
    write_reports(config, result, directory)
print("Both APR configurations passed with a test-only provider; no model API calls.")
PY
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$work:/work" --entrypoint python "$image" -c '
import json
from pathlib import Path
directories = list(Path("runs").iterdir())
assert len(directories) == len(list(Path("examples").glob("*.yaml"))) + 2
for directory in directories:
    for name in ("config.yaml", "manifest.json", "judgements.jsonl", "results.json", "report.md", "console.txt"):
        assert (directory / name).stat().st_size > 0, (directory, name)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["counts"]["failed_judgements"] == 0, manifest["counts"]
print("All example configurations produced valid results and persisted reports.")
'
