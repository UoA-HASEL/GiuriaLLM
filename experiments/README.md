# Paper experiments

The paper evaluates 100 items from each of four benchmarks across three runs. This directory contains configurations and scripts. Download the data and saved results from [Zenodo](https://zenodo.org/records/23584777).

| Benchmark | Configuration | Task | Aggregation |
| --- | --- | --- | --- |
| QuRE | [qure-listwise-panel.yaml](configs/qure-listwise-panel.yaml) | Rank requirements | Majority vote |
| CoNaLa | [conala-pointwise-panel.yaml](configs/conala-pointwise-panel.yaml) | Score code usefulness | Mean |
| CodeJudge-Bench | [cjb-testgen-panel.yaml](configs/cjb-testgen-panel.yaml) | Select the correct test | Majority vote |
| APR-Assess | [apr-pointwise-panel.yaml](configs/apr-pointwise-panel.yaml) | Classify patch correctness | Majority vote |

## Read saved results

Install GiuriaLLM using the [main README](../README.md), then extract the Zenodo archive. These commands read saved results without API calls:

```bash
archive=/path/to/GiuriaLLM-Evaluation
giuriallm show "$archive/experiments/runs/conala-rep1"
giuriallm summarize \
  "$archive/experiments/runs/conala-rep1" \
  "$archive/experiments/runs/conala-rep2" \
  "$archive/experiments/runs/conala-rep3" \
  -o experiments/summaries/conala-pointwise.json
```

Use the corresponding directories for other benchmarks. Summaries report mean and sample standard deviation; Human baselines remain fixed. See [reports and metrics](../docs/usage.md#reports-and-metrics).

## Run new experiments

From the repository root, copy the sampled data and check the configurations:

```bash
cp -R "$archive/experiments/data" experiments/data
for config in experiments/configs/*-panel.yaml; do
  giuriallm validate "$config" --check-dataset
done
python experiments/scripts/check_panel_settings.py
```

These checks do not call models. Review the recorded model IDs, endpoints, and prices, then set `OPENAI_API_KEY` and `AIHUBMIX_API_KEY` in your environment.

```bash
bash experiments/scripts/run_panel.sh rep1
bash experiments/scripts/run_panel.sh rep2
bash experiments/scripts/run_panel.sh rep3
```

Each command runs all four tasks and incurs API charges. To run one task, append its name, for example `bash experiments/scripts/run_panel.sh rep1 conala-pointwise`. Results are saved under `experiments/runs/`. Each repetition also writes a `.log` and an `.ids` file listing its run directories.

## Optional data preparation

The archived samples are ready to use. To rebuild them, install `pip install -e ".[validation]"` and use the scripts in [preparation/](preparation/). Their command-line options specify source files and output paths. [Source-check scripts](scripts/) compare samples with upstream data; the APR oracle requires a separate Defects4J environment.

Sources include [QuRE](https://zenodo.org/records/16049822), [CodeJudge-Bench](https://huggingface.co/datasets/mattymchen/codejudgebench), and [SE-Jury](https://github.com/Xin-Zhou-smu/SE-Jury). CoNaLa grades originate from [Out of the BLEU](https://github.com/JetBrains-Research/codegen-metrics); APR reference patches use [Defects4J](https://github.com/rjust/defects4j).

Downloaded data and generated results are ignored by Git. Cite the original datasets and annotation studies when reusing them.
