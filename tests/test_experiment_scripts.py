"""Check the experiment launcher without invoking a provider."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "run_panel.sh"
TASKS = ("conala-pointwise", "apr-pointwise", "qure-listwise", "cjb-testgen")


@pytest.fixture
def launcher(tmp_path):
    if not shutil.which("bash"):
        pytest.skip("bash is required for the experiment launcher")
    root = tmp_path / "experiments"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    shutil.copyfile(SOURCE, root / "scripts" / "run_panel.sh")
    for task in TASKS:
        (root / "configs" / f"{task}-panel.yaml").touch()
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "giuriallm"
    stub.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        'test "$1" = run\ntest "$3" = --quiet\n'
        'test "${HTTPS_PROXY-unset}" = "${EXPECTED_PROXY-unset}"\n'
        'if [ "${FAIL_RUN:-0}" = 1 ]; then exit 7; fi\n'
        "task=${2#configs/}\ntask=${task%-panel.yaml}\n"
        'mkdir -p "runs/$task"\ntouch "runs/$task/manifest.json"\n'
        'echo "runs/$task"\n'
    )
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
    env.update(OPENAI_API_KEY="test-only", AIHUBMIX_API_KEY="test-only")
    env.pop("HTTPS_PROXY", None)
    env.pop("EXPECTED_PROXY", None)
    env.pop("FAIL_RUN", None)

    def run(*args):
        return subprocess.run(
            ["bash", str(root / "scripts" / "run_panel.sh"), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

    return root, env, run


@pytest.mark.parametrize("proxy", [None, "http://example.invalid:8000"])
def test_panel_launcher_creates_runs_and_respects_proxy(launcher, proxy):
    root, env, run = launcher
    if proxy:
        env.update(HTTPS_PROXY=proxy, EXPECTED_PROXY=proxy)
    result = run("rep1")
    assert result.returncode == 0, result.stderr
    assert (root / "runs" / "panel-rep1.ids").read_text().splitlines() == [
        f"{task}\truns/{task}" for task in TASKS
    ]
    assert not list((root / "runs").glob(".panel-*"))


def test_panel_launcher_preserves_previous_mapping_on_failure(launcher):
    root, env, run = launcher
    assert run("rep1").returncode == 0
    ids = root / "runs" / "panel-rep1.ids"
    before = ids.read_bytes()
    env["FAIL_RUN"] = "1"
    assert run("rep1", "conala-pointwise").returncode == 7
    assert ids.read_bytes() == before
    assert not list((root / "runs").glob(".panel-*"))


def test_panel_launcher_replaces_only_requested_task(launcher):
    root, _, run = launcher
    assert run("rep1").returncode == 0
    ids = root / "runs" / "panel-rep1.ids"
    ids.write_text(ids.read_text().replace("runs/conala-pointwise", "runs/old-conala"))
    assert run("rep1", "conala-pointwise").returncode == 0
    assert dict(line.split("\t") for line in ids.read_text().splitlines()) == {
        task: f"runs/{task}" for task in TASKS
    }


@pytest.mark.parametrize("args", [("../rep1",), ("rep1", "unknown-task")])
def test_panel_launcher_rejects_invalid_arguments_before_running(launcher, args):
    root, _, run = launcher
    assert run(*args).returncode != 0
    assert not (root / "runs").exists()


def test_cjb_check_normalizes_serialized_string_outputs():
    source = SOURCE.with_name("verify_cjb_testgen.py").read_text()
    # Load the pure comparison helper without running the command-line script.
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "loose"
    )
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<helper>", "exec"), namespace)
    compare = namespace["loose"]
    assert compare('"hhhiii"') == compare("hhhiii")
    assert compare("[1, 2]") == compare("[1,2]")
    assert compare("hhhiii") != compare("nnnnnn")
