"""GiuriaLLM: a configuration-driven framework for LLM-as-a-Judge evaluation of
software artifacts.

An evaluation is a declarative artifact, not bespoke glue code. The
configuration schema encodes the LLM-as-a-Judge formalism

    E(T, C, X, R) -> (Y, E, F)

and bias mitigation and meta-evaluation of the judge are stages of the pipeline
rather than per-study scripts.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dataset import DatasetError, load_human_labels, load_items  # noqa: E402
from .engine import Engine, RunResult, run_config, write_run  # noqa: E402
from .spec import ConfigError, RunConfig, config_hash, load_config  # noqa: E402
from .types import EvaluationItem, ItemResult, Judgement, Usage, Verdict  # noqa: E402

__all__ = [
    "ConfigError",
    "DatasetError",
    "Engine",
    "EvaluationItem",
    "ItemResult",
    "Judgement",
    "RunConfig",
    "RunResult",
    "Usage",
    "Verdict",
    "__version__",
    "config_hash",
    "load_config",
    "load_human_labels",
    "load_items",
    "run_config",
    "write_run",
]
