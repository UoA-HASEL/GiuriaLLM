# GiuriaLLM

GiuriaLLM is a configurable LLM-as-a-judge framework for software engineering. It evaluates software artifacts using one LLM or a panel, with tasks, criteria, models, and input data defined in YAML.

- **Evaluation:** pointwise scoring, pairwise comparison, and listwise ranking or selection, with verdicts and resource-use reports.
- **Order sensitivity:** measures how often verdicts change when candidates are reordered.
- **Human alignment:** compares judge and human label distributions and reports results by human agreement, when annotations are available.

Supported providers include Anthropic, OpenAI, OpenAI-compatible endpoints, and local models served by Ollama.

<p align="center">
  <img src="docs/assets/overview.png" alt="GiuriaLLM overview: inputs, judging workflow, and outputs" width="480">
</p>

## Quick start with Docker

The image includes an APR-Assess example with the bug context, a candidate patch, a correct reference patch, and five human annotations. You need Docker. Both options below print a report in the terminal and save output files under `runs/`.

### Try without an API key

Run the example with simulated judge responses to check the workflow. No API key is required, and no model calls are made. Mock verdicts and resource figures are for testing only.

```bash
mkdir -p runs
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/runs:/work/runs" \
  ghcr.io/uoa-hasel/giuriallm:latest \
  run /opt/giuriallm/demo/configs/apr-patch.yaml --provider mock --no-cache
```

### Run with OpenAI

Evaluate the same patch with OpenAI's `gpt-5.4-mini`. The model scores the patch as 0 (incorrect) or 1 (correct), and GiuriaLLM compares its verdict with the human annotations. You need an OpenAI API key. API usage is billed to your account.

```bash
mkdir -p runs
export OPENAI_API_KEY="your-api-key"
docker run --rm --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD/runs:/work/runs" \
  ghcr.io/uoa-hasel/giuriallm:latest \
  run /opt/giuriallm/demo/configs/apr-patch.yaml
```

These commands work in a macOS or Linux terminal. The [usage guide](docs/usage.md) covers YAML configuration, panels, custom endpoints, PowerShell, and the Python API. Versioned image tags are published with GitHub releases.

## Results

Each run writes a timestamped directory:


| File                        | Contents                                        |
| --------------------------- | ----------------------------------------------- |
| `report.md` / `console.txt` | Evaluation and resource-use report              |
| `results.json`              | Summary metrics and each item's final result    |
| `judgements.jsonl`          | Individual model judgements, including failures |
| `config.yaml`               | Resolved run configuration                      |
| `manifest.json`             | Run metadata, fingerprints, and resource totals |


Reports show each judge and, when configured, the panel. Resource tables report tokens, estimated cost, cumulative API time, and estimated energy per input item.

Use `giuriallm show RUN_DIR` to read saved results and `giuriallm summarize RUN_DIR...` to combine repeated runs. Neither command calls a model. See [reports and metrics](docs/usage.md#reports-and-metrics) for definitions.

## Examples and paper data


| Configuration                                             | Purpose                                                   |
| --------------------------------------------------------- | --------------------------------------------------------- |
| [APR pointwise](demo/configs/apr-patch.yaml)              | Evaluate one patch with OpenAI                            |
| [APR five-judge panel](demo/configs/apr-five-judges.yaml) | Original configuration used in the paper demo             |
| [Pairwise](examples/code-gen-pairwise.yaml)               | Compare two implementations and measure order sensitivity |
| [Listwise](examples/code-gen-listwise.yaml)               | Rank three implementations across candidate permutations  |


Both APR examples use the same [patch input](demo/data/apr-patch.jsonl) and [human annotations](demo/data/human-labels.jsonl). The five-judge example calls models through AiHubMix. To run it with Docker, set `AIHUBMIX_API_KEY` to your AiHubMix API key, pass it with `-e AIHUBMIX_API_KEY`, and use the `apr-five-judges.yaml` configuration.

In the recorded five-judge run, three judges were correct, two were incorrect, and the panel's majority verdict was correct. New runs may differ. The single-judge quick start does not reproduce that panel setup.

The [Zenodo archive](https://zenodo.org/records/23584777) contains the evaluation data, recorded demo, and experiment results. See [paper experiments](experiments/README.md) for configurations and reproduction commands.

## Local installation

You need Python 3.11 or newer. Run these commands in a macOS or Linux terminal:

```bash
git clone https://github.com/UoA-HASEL/GiuriaLLM.git
cd GiuriaLLM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export OPENAI_API_KEY="your-api-key"
giuriallm validate demo/configs/apr-patch.yaml --check-dataset
giuriallm run demo/configs/apr-patch.yaml
```

For development, install `pip install -e ".[dev]"`, then run `pytest` and `ruff check src tests`. To build a local image, run `docker build -t giuriallm:local .`.

## Paper and authors

**GIURIALLM: A Configurable LLM-as-a-Judge Framework for Software Engineering**

Yeheng Chen and Valerio Terragni, University of Auckland.

Under review at the ICSE 2027 Tool Demonstration and Data Showcase Track.

## License

GiuriaLLM is released under the [Apache License 2.0](LICENSE). Benchmark data retain their original licenses.
