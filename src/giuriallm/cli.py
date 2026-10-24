"""Command-line interface.

giuriallm run demo/configs/apr-patch.yaml
giuriallm validate demo/configs/apr-patch.yaml
giuriallm show runs/<run-id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import DatasetError, load_items
from .engine import run_config, write_run
from .engine.manifest import read_manifest
from .evidence import available_evidence
from .providers import available_providers
from .report import render_report, write_reports
from .report.repetitions import (
    RepetitionError,
    render_repetitions,
    summarize_runs,
    write_repetitions,
)
from .spec import ConfigError, Metric, RunConfig, config_hash, load_config

EXAMPLE_CONFIG = """# A minimal GiuriaLLM evaluation. Every key maps to the LLM-as-a-Judge
# formalism E(T, C, X, R) -> (Y, E, F).
judge:
  type: pointwise                # T
  criteria:                      # C
    - name: functional_correctness
      definition: "Does the code produce the expected output for the task description?"
      scale: {kind: likert, min: 1, max: 5}
  reference: {mode: none}        # R
  outputs:                       # Y, E, F
    result: graded
    explanation: true
    feedback: false

model:
  provider: anthropic
  id: claude-opus-5
  samples: 1

prompt:
  strategy: cot
  order: explain_then_rate

debias:
  position_swap: false

dataset:
  path: data/items.jsonl
  fields:
    id: id
    context: prompt
    candidates: [candidate]
"""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except (ConfigError, DatasetError, RepetitionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="giuriallm",
        description="Configuration-driven LLM-as-a-Judge evaluation of software artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"giuriallm {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run an evaluation from a config file")
    run.add_argument("config", type=Path)
    run.add_argument("--limit", type=int, help="judge at most N items")
    run.add_argument("--provider", help="override model.provider (e.g. 'mock' for a dry run)")
    run.add_argument("--model", help="override model.id")
    run.add_argument("--samples", type=int, help="override model.samples")
    run.add_argument("--no-cache", action="store_true", help="ignore the response cache")
    run.add_argument("-o", "--output", type=Path, help="override report.output_dir")
    run.add_argument("-q", "--quiet", action="store_true", help="print only the run directory")
    run.add_argument(
        "-v", "--verbose", action="store_true", help="also print detailed report fields"
    )
    run.set_defaults(handler=_cmd_run)

    validate = sub.add_parser("validate", help="validate a config and print its hash")
    validate.add_argument("config", type=Path)
    validate.add_argument("--check-dataset", action="store_true", help="also load the dataset")
    validate.set_defaults(handler=_cmd_validate)

    show = sub.add_parser("show", help="summarise a completed run directory")
    show.add_argument("run_dir", type=Path)
    show.add_argument("--json", action="store_true", help="print the raw manifest")
    show.add_argument(
        "-v", "--verbose", action="store_true", help="also print detailed report fields"
    )
    show.set_defaults(handler=_cmd_show)

    summarize = sub.add_parser("summarize", help="summarise repeated runs (mean and sample SD)")
    summarize.add_argument("run_dirs", type=Path, nargs="+")
    summary_format = summarize.add_mutually_exclusive_group()
    summary_format.add_argument("--json", action="store_true", help="print machine-readable JSON")
    summary_format.add_argument("--markdown", action="store_true", help="print a Markdown report")
    summarize.add_argument(
        "-o", "--output", type=Path, help="save JSON plus companion Markdown and console reports"
    )
    summarize.set_defaults(handler=_cmd_summarize)

    init = sub.add_parser("init", help="write a starter config")
    init.add_argument("path", type=Path, nargs="?", default=Path("giuriallm.yaml"))
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=_cmd_init)

    plugins = sub.add_parser("plugins", help="list registered providers, evidence, and metrics")
    plugins.set_defaults(handler=_cmd_plugins)

    return parser


# --------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = _apply_overrides(load_config(args.config), args)

    result = asyncio.run(run_config(config))

    run_dir = write_run(config, result, config.report.output_dir)
    write_reports(config, result, run_dir)

    if args.quiet:
        print(run_dir)
        return 0

    print(render_report(config, result, verbose=args.verbose), end="")
    print(f"Reports saved to: {run_dir}")
    return 1 if result.n_failed else 0


def _apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    data = json.loads(config.model_dump_json())
    if args.limit is not None:
        data["dataset"]["limit"] = args.limit
    if args.provider:
        data["model"]["provider"] = args.provider
        for member in data.get("panel", {}).get("members", []) if data.get("panel") else []:
            member["provider"] = args.provider
    if args.model:
        data["model"]["id"] = args.model
    if args.samples is not None:
        data["model"]["samples"] = args.samples
    if args.no_cache:
        data["engine"]["cache"] = False
    if args.output is not None:
        data["report"]["output_dir"] = str(args.output)
    return RunConfig.model_validate(data)


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"ok: {args.config}")
    print(f"  config hash: {config_hash(config)}")
    print(f"  {config.judge.type.value} -> {config.judge.outputs.result.value}")
    print(f"  criteria: {', '.join(c.name for c in config.judge.criteria)}")
    print(f"  judges: {', '.join(m.display for m in config.judges())}")
    if args.check_dataset:
        items = load_items(config.dataset, config.judge.reference.mode)
        print(f"  dataset: {len(items)} items from {config.dataset.path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.run_dir)
    if args.json:
        print(json.dumps(manifest, indent=2))
        return 0
    from .report.saved import saved_manifest
    from .report.summary import render_summary

    manifest = saved_manifest(args.run_dir)
    print(render_summary(manifest), end="")
    if args.verbose:
        from .report import _render_detailed

        print(
            _render_detailed(manifest.get("evaluations", {}), manifest.get("human_baseline")),
            end="",
        )
    print(f"Reports saved to: {args.run_dir}")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    summary = summarize_runs(args.run_dirs)
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        write_repetitions(summary, args.output)
    print(payload.rstrip() if args.json else render_repetitions(summary, markdown=args.markdown))
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    if args.path.exists() and not args.force:
        print(f"error: {args.path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    args.path.write_text(EXAMPLE_CONFIG)
    print(f"wrote {args.path}")
    print("next: point dataset.path at a JSONL file, then run 'giuriallm validate'")
    return 0


def _cmd_plugins(_: argparse.Namespace) -> int:
    print("providers: " + ", ".join(available_providers()))
    print("evidence:  " + ", ".join(available_evidence()))
    print("metrics:   " + ", ".join(m.value for m in Metric))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
