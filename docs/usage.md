# Usage

Start with the [APR configuration](../demo/configs/apr-patch.yaml). GiuriaLLM reads YAML and a JSON Lines dataset. Run `giuriallm validate CONFIG --check-dataset` to check both before calling a model.

This guide covers [configuration](#task-and-criteria), [Docker](#docker), [Python](#python-api), and [reports](#reports-and-metrics).

## Task and criteria

| `judge.type` | Candidate fields | `judge.outputs.result` |
| --- | --- | --- |
| `pointwise` | One | `graded` |
| `pairwise` | Two | `best_choice` |
| `listwise` | Three or more | `ranking` or `best_choice` |

Each entry in `judge.criteria` has a unique `name`, a `definition`, and a `scale`:

```yaml
judge:
  type: pointwise
  criteria:
    - name: correctness
      definition: Does the implementation satisfy the task, including edge cases?
      scale: {kind: binary}
  reference: {mode: required}
  outputs:
    result: graded
    explanation: false
    feedback: false
```

Scales can be `binary` (0 or 1), `likert` (integer scores with `min` and `max`), `continuous` (numeric scores with `min` and `max`), or `categorical` (a list of `categories`). Numeric scales can include named `anchors`. Reference mode is `none`, `optional`, or `required`. Explanations and improvement feedback are enabled independently.

## Input data

An item may use your existing field names:

```json
{"id":"example-1","prompt":"Return the larger integer.","candidate":"def larger(a, b): return max(a, b)","reference":"def larger(a, b): return a if a > b else b"}
```

Map those names in the configuration:

```yaml
dataset:
  path: data/items.jsonl
  fields:
    id: id
    context: prompt
    candidates: [candidate]
    reference: reference
```

For pairwise and listwise tasks, list the candidate fields in their original order. Choice labels use zero-based indices into that list. Dataset, human-label, and few-shot-example paths are relative to the configuration file.

## Models and panels

Use `model` for a single judge. For the OpenAI API, set `OPENAI_API_KEY` in your environment:

```yaml
model:
  provider: openai
  id: gpt-5.4-mini
  temperature: null
  max_tokens: 32768
  samples: 1
  price_per_mtok: [0.75, 4.50]
  options:
    structured_output: json_schema
    seed: false
```

No custom endpoint is needed for OpenAI. The example uses [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini); change the model ID and token prices to use another model.

`temperature: null` leaves the sampling temperature unset. `samples` controls repeated judgements for each presentation order. `structured_output` accepts `tools` or `json_schema`; `seed: false` omits the seed parameter.

For an OpenAI-compatible service, keep `provider: openai`, use that service's model ID and prices, and add its endpoint and credential variable under `options`:

```yaml
  options:
    base_url: https://your-endpoint.example/v1
    api_key_env: LLM_API_KEY
    structured_output: json_schema
    seed: false
```

Set `LLM_API_KEY` in your environment before running. With Docker, pass it using `-e LLM_API_KEY`. Provider-specific options are described in the [OpenAI](../src/giuriallm/providers/openai_provider.py), [Anthropic](../src/giuriallm/providers/anthropic_provider.py), and [Ollama](../src/giuriallm/providers/ollama_provider.py) adapters.

| Provider | Default credential or service |
| --- | --- |
| `openai` | `OPENAI_API_KEY`; custom endpoints can use `options.api_key_env` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | Ollama at `http://localhost:11434` |
| `mock` | No credentials; deterministic test judgements |

For Ollama, set `provider: ollama` and use the ID of an installed model. Set `options.base_url` if the service is not at its default address.

To use a panel, replace `model` with `panel`. Each member has its own model, provider, and settings. For example, this template combines OpenAI and Anthropic judges:

```yaml
panel:
  aggregation: majority_vote
  members:
    - provider: openai
      id: gpt-5.4-mini
      temperature: null
      max_tokens: 32768
      price_per_mtok: [0.75, 4.50]
      samples: 1
      options:
        structured_output: json_schema
        seed: false
    - provider: anthropic
      id: your-anthropic-model-id
      samples: 1
```

Replace the Anthropic model ID and set both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. With Docker, pass both using `-e OPENAI_API_KEY -e ANTHROPIC_API_KEY`. You can add more members, including models at custom endpoints.

Use `majority_vote` for discrete verdicts or `mean` for numeric ratings. Numeric ratings also support `median` and `weighted`. `panel.weights` supplies weights keyed by judge label for weighted aggregation. Panel aggregation reuses member responses and makes no additional model call.

Set `price_per_mtok: [input_price, output_price]` on each model to specify USD per million tokens. These prices are used for cost estimates.

## Prompting and candidate order

`prompt.strategy` accepts `direct`, `cot`, `rubric`, `fewshot`, or `criteria_decomposed`. Few-shot prompting requires `prompt.examples`, a YAML file containing examples. `prompt.order` accepts `explain_then_rate` or `rate_then_explain`.

Pairwise tasks swap candidates by default. Set `debias.position_swap: false` to use only the original order. For listwise tasks, set `debias.permutations` above 1 to evaluate additional orders. The report measures verdict changes across valid orders when `debias.report_flip_rate` is enabled.

## Human annotations

Supply annotations separately from reference artifacts. They are used to evaluate the judges and are not included in judge prompts.

```json
{"id":"example-1","labels":[1,1,0],"annotators":["rater-a","rater-b","rater-c"]}
```

```yaml
meta_evaluation:
  human_labels: data/human-labels.jsonl
  metrics: [error, jensen_shannon]
  stratify_by_agreement: true
  agreement_threshold: 0.8
```

Annotator IDs are optional; include them to compute the leave-one-annotator-out Human baseline. IDs must remain consistent across items. A single reference label can be supplied as `{"id":"example-1","label":1}`. See [reports and metrics](#reports-and-metrics) for metric definitions.

## Execution and output

```yaml
engine:
  concurrency: 5
  max_retries: 3
  cache: true
  cache_dir: .giuriallm-cache
  energy_wh_per_ktok: 0.4
report:
  output_dir: runs
  formats: [markdown, json]
```

Output and cache paths are relative to the working directory. `--no-cache` disables local response reuse for a run. `--limit N` limits the input items; `-o DIR` changes the output directory. HTML and LaTeX reports can also be requested through `report.formats`.

Optional `evidence` plugins add information before judgement. `python_syntax` checks syntax without executing code. The `command` plugin executes candidate code and should be used in an isolated environment without credentials.

## Docker

To edit the bundled example, copy it into an empty working directory:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" --entrypoint cp \
  ghcr.io/uoa-hasel/giuriallm:latest -R /opt/giuriallm/demo /work/demo
```

Edit `demo/configs/apr-patch.yaml`, then run:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD:/work" \
  ghcr.io/uoa-hasel/giuriallm:latest \
  run demo/configs/apr-patch.yaml
```

For other providers, pass their credential variables with `-e`. For your own data, keep the configuration and input files under the mounted directory and use its relative path after `run`. Replace `run CONFIG` with `validate CONFIG --check-dataset` to check inputs, or `show runs/RUN_ID` to read saved results.

On Linux and macOS, `--user` preserves file ownership. The image otherwise runs as UID 1000. Create output directories before mounting them. Use a release tag or image digest instead of `latest` when recording an environment.

For Ollama running on the host, set `options.base_url: http://host.docker.internal:11434`. On Linux, add `--add-host=host.docker.internal:host-gateway`. Ollama must listen on an address reachable from the container and have the requested model installed.

### PowerShell

With Docker Desktop running Linux containers:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$env:OPENAI_API_KEY = "your-api-key"
docker run --rm -e OPENAI_API_KEY -v "${PWD}/runs:/work/runs" `
  ghcr.io/uoa-hasel/giuriallm:latest `
  run /opt/giuriallm/demo/configs/apr-patch.yaml
```

## Python API

```python
import asyncio

from giuriallm.engine import run_config, write_run
from giuriallm.report import write_reports
from giuriallm.spec import load_config

config = load_config("demo/configs/apr-patch.yaml")
result = asyncio.run(run_config(config))
run_dir = write_run(config, result, config.report.output_dir)
write_reports(config, result, run_dir)
print(run_dir)
```

This uses the same configuration, engine, and reporting code as the CLI.

## Reports and metrics

Console and Markdown reports share five sections: Run summary, Judges, Evaluation, Resource use, and Notes. Metric rows follow `meta_evaluation.metrics`. Human-agreement groups and order sensitivity depend on the annotations and candidate orders available. `n/a` means undefined or unavailable; `--` means not reported.

Coverage counts input items, valid results, and failed judgements. **Number of items** counts the items included in that table's evaluation. Counts can differ between judges when requests fail or reference labels are missing.

| Metric | Meaning |
| --- | --- |
| `error` | Mean absolute error normalized by the score range; label mismatch for binary or categorical outcomes |
| `accuracy` | Fraction of matching labels |
| `cohen_kappa` / `krippendorff_alpha` | Agreement with reference labels |
| `spearman` / `kendall` | Rank correlation with human ratings |
| `icc` | Intraclass correlation |
| `earth_movers_distance` | Distance between ordered score distributions |
| `jensen_shannon` | Divergence between label distributions |

Numeric human targets use the mean annotation. Binary targets use majority vote, with ties assigned to 1. Distribution metrics retain the annotation distribution. Metrics such as correlation may be undefined for a single item.

**Human** is a leave-one-annotator-out baseline: each annotator is evaluated against the others. It requires annotator IDs; `n` counts contributing annotators. High and low agreement refer to human agreement under the configured grouping rules. A single label does not establish human agreement.

**Flip rate** measures how often the selected candidate changes across presentation orders, after mapping choices back to the original order. It is computed for each judge and the panel. An item needs at least two valid orders to contribute.

For majority-vote choice tasks, votes are combined within each order and then across orders. Within-order ties follow member order; across-order ties prefer the original-order choice when it is tied. For majority-vote rankings, the panel records the top candidate; individual rankings remain in `judgements.jsonl`.

Resource values are averaged over all input items, including failed ones. Panel values sum its members. Cumulative API time sums call durations across orders and samples; it is not wall-clock time. Cost uses configured token prices. Energy uses `engine.energy_wh_per_ktok` and is estimated, not measured. JSON outputs retain totals.

### Saved reports

```bash
giuriallm show runs/RUN_ID
giuriallm summarize runs/RUN_1 runs/RUN_2 runs/RUN_3 -o summaries/task.json
```

`show` reads saved results without changing files. `summarize` writes JSON, `.report.md`, and `.console.txt`, using the mean and sample standard deviation across matching runs. Configuration and data fingerprints must match; identical Human baselines remain fixed. Neither command calls a model.

To regenerate reports:

```python
from pathlib import Path
from giuriallm.report.saved import regenerate_reports

regenerate_reports(Path("runs/RUN_ID"))
```

If annotations have moved, add `labels_path=Path("data/human-labels.jsonl")`; the file must match the original fingerprint. Regeneration updates derived reports and results, preserving the configuration, manifest, and individual judgements.
