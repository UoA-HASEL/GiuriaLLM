# Paper experiments

The paper evaluates 100 items from each of four benchmarks across three repetitions. This directory contains the configurations and scripts. Data, individual judgements, reports, and three-run summaries are available in the [Zenodo archive](https://zenodo.org/records/23584777).

| Benchmark | Configuration | Task | Panel aggregation |
| --- | --- | --- | --- |
| QuRE | `configs/qure-listwise-panel.yaml` | Rank five requirements; evaluate the top choice | Majority vote |
| CoNaLa | `configs/conala-pointwise-panel.yaml` | Score code usefulness from 0 to 4 | Mean |
| CodeJudge-Bench | `configs/cjb-testgen-panel.yaml` | Select the correct test | Majority vote |
| APR-Assess | `configs/apr-pointwise-panel.yaml` | Classify patch correctness | Majority vote |

## Prepare

Install GiuriaLLM using the [main README](../README.md). Download and extract the evaluation archive, then copy the sampled inputs from its `experiments/data` directory. From the repository root:

```bash
archive=/path/to/GiuriaLLM-Evaluation
cp -R "$archive/experiments/data" experiments/data
for config in experiments/configs/*-panel.yaml; do
  giuriallm validate "$config" --check-dataset
done
python experiments/scripts/check_panel_settings.py
```

These commands check data loading and request settings without calling models. The settings check writes its inspection to `experiments/runs/settings-check-*/`.

## Run new experiments

The configurations record the models, endpoints, inference settings, and prices used in the paper. Review them before a new run and provide `OPENAI_API_KEY` and `AIHUBMIX_API_KEY` in your environment.

From the repository root:

```bash
bash experiments/scripts/run_panel.sh rep1
bash experiments/scripts/run_panel.sh rep2
bash experiments/scripts/run_panel.sh rep3
```

Each command calls the models for all four tasks. To run one task, append its name, for example `bash experiments/scripts/run_panel.sh rep1 conala-pointwise`. Outputs are saved under `experiments/runs/`. Each repetition also writes a `.log` and a `.ids` file mapping task names to run directories. Use different repetition labels for concurrent launches.

## Read the saved results

Read or summarize the archived runs without making API calls:

```bash
giuriallm show "$archive/experiments/runs/conala-rep1"
giuriallm summarize \
  "$archive/experiments/runs/conala-rep1" \
  "$archive/experiments/runs/conala-rep2" \
  "$archive/experiments/runs/conala-rep3" \
  -o experiments/summaries/conala-pointwise.json
```

Use the corresponding run directories for the other benchmarks. Summaries report mean and sample standard deviation; Human baselines remain fixed. See [reports and metrics](../docs/usage.md#reports-and-metrics) for definitions.

## Rebuild sampled inputs

`preparation/` contains the sampling and reference-patch conversion scripts. Install their additional dependencies with `pip install -e ".[validation]"` from the repository root, then place the upstream files under `experiments/raw/`:

| Source | Paths relative to `experiments/` |
| --- | --- |
| [QuRE](https://zenodo.org/records/16049822) | `raw/QuRE/QuRE.csv` |
| [CodeJudge-Bench TestGen](https://huggingface.co/datasets/mattymchen/codejudgebench) | `raw/CodeJudgeBench/testgen/gemini_2.5_flash-00000-of-00001.parquet` and `gemini_2.5_pro-00000-of-00001.parquet` |
| [SE-Jury](https://github.com/Xin-Zhou-smu/SE-Jury), CoNaLa | `raw/SE-Jury/conala/conala-human-grades.json` and `conala_grade.json` |
| SE-Jury, APR | `raw/SE-Jury/apr/bach_data.pkl` and `bach_label.csv` |
| [Defects4J](https://github.com/rjust/defects4j) | `raw/SE-Jury/apr/d4j_gold_patches.json`, generated below |

CoNaLa human grades originate from [Out of the BLEU](https://github.com/JetBrains-Research/codegen-metrics). SE-Jury distributes the CoNaLa and APR files in `se_jury.zip`, under `src/self_evaluation_loop_flow/data/`.

From `experiments/`:

```bash
python preparation/d4j_gold_patches.py --patches /path/to/d4j-patches
python preparation/build_qure_listwise.py
python preparation/build_conala_pointwise.py
python preparation/build_cjb_testgen.py
python preparation/build_apr_pointwise.py
```

The Defects4J source directory must contain `<Project>/<bug>.src.patch` files. The converter reverses fixed-to-buggy diffs to produce reference patches. Only load trusted pickle files.

The builders use seed 0 and task-specific sampling rules: weak-word groups for QuRE, agreement and generation-system groups for CoNaLa, recent distinct questions from two Gemini splits for CodeJudge-Bench, and agreement/majority-label strata for APR. Use `--out-prefix` to write rebuilt samples to a separate location.

## Source checks

From `experiments/`:

- `python scripts/verify_qure.py` compares sampled requirements with the original CSV.
- `python scripts/verify_cjb_testgen.py --lcb DIR` compares test outputs with LiveCodeBench `test5.jsonl` and `test6.jsonl`.
- `bash scripts/run_apr_oracle.sh 8` checks patches against trigger tests in an external `giuriallm-d4j` image. This image must contain Python 3, Java, `patch`, and an initialized Defects4J installation. It is separate from the GiuriaLLM runtime image.

Data downloads, rebuilt samples, and new runs are ignored by Git. Cite the original datasets and annotation studies when reusing them.
