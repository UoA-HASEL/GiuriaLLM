"""Tool-augmented judging: evidence gathered *before* the verdict.

Each plugin inspects an item's candidates with real tooling and returns a
structured result that the prompt compiler renders into the prompt. The judge
then reasons over observed behaviour rather than over source text alone.

Plugins are configured by name in the ``evidence`` section of a run config.
"""

from __future__ import annotations

import ast
import asyncio
import shlex
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..spec import EvidencePlugin
from ..types import EvaluationItem


class EvidenceError(RuntimeError):
    pass


class EvidenceProvider(ABC):
    name: str

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = options or {}

    @abstractmethod
    async def gather(self, item: EvaluationItem) -> Any:
        """Return evidence about the item's candidates."""


class PythonSyntaxEvidence(EvidenceProvider):
    """Parse each candidate as Python. Static only -- nothing is executed."""

    name = "python_syntax"

    async def gather(self, item: EvaluationItem) -> Any:
        out = []
        for index, candidate in enumerate(item.candidates):
            try:
                ast.parse(candidate)
                out.append({"candidate": index, "parses": True})
            except SyntaxError as exc:
                out.append(
                    {
                        "candidate": index,
                        "parses": False,
                        "error": f"line {exc.lineno}: {exc.msg}",
                    }
                )
        return out


class CommandEvidence(EvidenceProvider):
    """Run a configured command against each candidate and capture its result.

    .. warning::
       This executes candidate code, which is model-generated and untrusted. Run
       it inside the project's container image or another isolated environment,
       never on a developer machine with ambient credentials.

    Options:
        ``command``   shell command; ``{file}`` is replaced with the candidate path
        ``filename``  name the candidate is written to (default ``candidate.py``)
        ``timeout``   seconds before the command is killed (default 30)
        ``files``     extra ``{name: content}`` files written beside the candidate
        ``meta_files`` item metadata keys whose values are written as files
    """

    name = "command"

    async def gather(self, item: EvaluationItem) -> Any:
        command = self.options.get("command")
        if not command:
            raise EvidenceError("evidence plugin 'command' requires an options.command string")
        filename = self.options.get("filename", "candidate.py")
        timeout = float(self.options.get("timeout", 30))

        results = []
        for index, candidate in enumerate(item.candidates):
            workdir = Path(tempfile.mkdtemp(prefix="giuriallm-evidence-"))
            try:
                target = workdir / filename
                target.write_text(candidate)
                for name, content in (self.options.get("files") or {}).items():
                    (workdir / name).write_text(str(content))
                for key in self.options.get("meta_files") or []:
                    if key in item.meta:
                        (workdir / key).write_text(str(item.meta[key]))
                results.append(
                    await _run(command.format(file=target.name), workdir, timeout, index)
                )
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        return results


async def _run(command: str, cwd: Path, timeout: float, index: int) -> dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise EvidenceError(f"command not found: {command!r} ({exc})") from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return {"candidate": index, "timed_out": True, "timeout_s": timeout}

    return {
        "candidate": index,
        "exit_code": process.returncode,
        "stdout": _tail(stdout.decode(errors="replace")),
        "stderr": _tail(stderr.decode(errors="replace")),
    }


def _tail(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


_REGISTRY: dict[str, Callable[[dict[str, Any]], EvidenceProvider]] = {
    PythonSyntaxEvidence.name: PythonSyntaxEvidence,
    CommandEvidence.name: CommandEvidence,
}


def register_evidence(name: str, factory: Callable[[dict[str, Any]], EvidenceProvider]) -> None:
    _REGISTRY[name] = factory


def build_evidence(plugins: list[EvidencePlugin]) -> list[EvidenceProvider]:
    providers: list[EvidenceProvider] = []
    for plugin in plugins:
        if plugin.name not in _REGISTRY:
            raise EvidenceError(
                f"unknown evidence plugin {plugin.name!r}; available: {sorted(_REGISTRY)}"
            )
        providers.append(_REGISTRY[plugin.name](plugin.options))
    return providers


def available_evidence() -> list[str]:
    return sorted(_REGISTRY)
