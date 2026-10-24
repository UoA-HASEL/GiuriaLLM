# Usage

Start with the [APR example](../demo/configs/apr-patch.yaml) and the [Docker quick start](../README.md#quick-start-with-docker). To use your own data or models, edit the YAML configuration as described below.

## Input and task

Input files use JSON Lines, with one item per line:

```json
{"id":"example-1","prompt":"Return the larger integer.","candidate":"def larger(a, b): return max(a, b)","reference":"def larger(a, b): return a if a > b else b"}
```

Map your field names in the configuration:

```yaml
dataset:
  path: data/items.jsonl
  fields:
    id: id
    context: prompt
    candidates: [candidate]
    reference: reference
```

Choose the task and output:

| `judge.type` | Candidates | `judge.outputs.result` |
| --- | --- | --- |
| `pointwise` | One | `graded` |
| `pairwise` | Two | `best_choice` |
| `listwise` | Three or more | `ranking` or `best_choice` |

Each criterion in `judge.criteria` has a name, definition, and scale. Supported scales are `binary`, `likert`, `continuous`, and `categorical`. References can be `none`, `optional`, or `required`. Enable explanations and feedback with `judge.outputs.explanation` and `judge.outputs.feedback`.

Dataset and annotation paths are relative to the YAML file. Output paths are relative to the working directory.

## Models and panels

The APR example uses one OpenAI model. Set `OPENAI_API_KEY` and edit `model.id` to change it. `samples` sets the number of judgements per presentation order. `max_tokens` limits output length. `temperature: null` leaves the sampling temperature unset.

| Provider | Credentials or service |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | An installed model served at `http://localhost:11434` |
| `mock` | No key needed. Returns simulated responses. |

For an OpenAI-compatible endpoint, keep `provider: openai`, use its model ID, and add:

```yaml
  options:
    base_url: https://your-endpoint.example/v1
    api_key_env: LLM_API_KEY
    structured_output: json_schema
    seed: false
```

Set `LLM_API_KEY` in your environment. Update `price_per_mtok: [input_price, output_price]` for cost estimates, in USD per million tokens.

For a panel, replace `model` with `panel`. Each member accepts the same model settings:

```yaml
panel:
  aggregation: majority_vote
  members:
    - provider: openai
      id: gpt-5.4-mini
      temperature: null
      max_tokens: 32768
      samples: 1
      options:
        structured_output: json_schema
        seed: false
    - provider: anthropic
      id: your-anthropic-model-id
      samples: 1
```

Replace the placeholder model ID, add token prices, and provide keys for both services. Add more members as needed. Use `majority_vote` for discrete verdicts or `mean` for numeric ratings. Aggregation makes no additional model call. The [five-judge APR configuration](../demo/configs/apr-five-judges.yaml) provides a complete panel example.

## Prompting and order

`prompt.strategy` supports `direct`, `cot`, `rubric`, `fewshot`, and `criteria_decomposed`. For few-shot prompting, set `prompt.examples` to a YAML file containing examples.

Pairwise tasks swap candidates by default. Set `debias.position_swap: false` to disable swapping. For listwise tasks, set `debias.permutations` above 1 to test multiple orders. Enable `debias.report_flip_rate` to report order sensitivity.

## Human annotations

Human labels are optional and are not sent to the judges. Supply a separate JSON Lines file using the same item IDs:

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

Include consistent annotator IDs to compute the Human baseline. A single reference label can use `{"id":"example-1","label":1}`.

## Run with Docker

Copy the bundled example into an empty working directory:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" --entrypoint cp \
  ghcr.io/uoa-hasel/giuriallm:latest -R /opt/giuriallm/demo /work/demo
```

Edit the configuration, then run it:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD:/work" \
  ghcr.io/uoa-hasel/giuriallm:latest \
  run demo/configs/apr-patch.yaml
```

Pass other credential variables with `-e`, such as `-e ANTHROPIC_API_KEY`. Keep custom configurations and data under the mounted directory. Use `validate CONFIG --check-dataset` in place of `run CONFIG` to check inputs without model calls.

`--limit N` limits the number of items, `-o DIR` sets the output directory, and `--no-cache` disables cached responses. For reproducible environments, use a versioned image tag or digest.

For Ollama on the host, set `options.base_url: http://host.docker.internal:11434`. Linux also needs `--add-host=host.docker.internal:host-gateway`. The service must be reachable from the container.

On Windows, use PowerShell with Docker Desktop running Linux containers:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$env:OPENAI_API_KEY = "your-api-key"
docker run --rm -e OPENAI_API_KEY -v "${PWD}/runs:/work/runs" `
  ghcr.io/uoa-hasel/giuriallm:latest `
  run /opt/giuriallm/demo/configs/apr-patch.yaml
```

## Python API

With a local installation, use the same configuration from Python:

```python
import asyncio

from giuriallm.engine import run_config, write_run
from giuriallm.report import write_reports
from giuriallm.spec import load_config

config = load_config("demo/configs/apr-patch.yaml")
result = asyncio.run(run_config(config))
run_dir = write_run(config, result, config.report.output_dir)
write_reports(config, result, run_dir)
```

## Reports and metrics

Reports contain Run summary, Judges, Evaluation, Resource use, and Notes. Select metrics through `meta_evaluation.metrics`:

| Metric | Meaning |
| --- | --- |
| `error` | Normalized absolute error, or the incorrect-label rate for binary or categorical outcomes |
| `accuracy` | Fraction of matching labels |
| `cohen_kappa` / `krippendorff_alpha` | Agreement with reference labels |
| `spearman` / `kendall` / `icc` | Correlation or agreement with human ratings |
| `earth_movers_distance` / `jensen_shannon` | Comparison of judge and human label distributions |

Coverage shows valid results and failures. **Number of items** counts the items evaluated in a table. **Human** compares each annotator with the others. `n` counts contributing annotators. Agreement groups refer to human agreement. **Flip rate** measures verdict changes across candidate orders. Notes explain unavailable values.

Resource use is averaged over all input items. Panel values sum its members. API time is cumulative, not wall-clock time. Cost and energy are estimates.

With a local installation, read or summarize saved results without calling models:

```bash
giuriallm show runs/RUN_ID
giuriallm summarize runs/RUN_1 runs/RUN_2 runs/RUN_3 -o summaries/task.json
```

Summaries report the mean and sample standard deviation across runs with matching configurations and data. With Docker, mount the results directory and pass the same `show` or `summarize` arguments after the image name.
