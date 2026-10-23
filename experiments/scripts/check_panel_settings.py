"""Inspect actual requests; --live additionally calls one item/order per task/model.

Run from GiuriaLLM with PYTHONPATH=src. No dataset, historical run, or paper is
modified. Each invocation creates a separate JSON report. Live mode disables
SDK retries and uses a 90-second request timeout; this is a compatibility check,
not an evaluation run. Credentials are read only from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from giuriallm.dataset import load_items
from giuriallm.debias import variants_for
from giuriallm.prompts import PromptCompiler, parse_verdict
from giuriallm.prompts.compiler import TOOL_NAME
from giuriallm.providers import LLMRequest, OpenAIProvider
from giuriallm.spec import config_hash, load_config


def safe_json(value, **kwargs):
    text = json.dumps(value, **kwargs)
    for name in ("OPENAI_API_KEY", "AIHUBMIX_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


class RecordingClient:
    def __init__(self, client, attempts):
        self.client = client
        self.attempts = attempts
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **params):
        attempt = {"request": params}
        self.attempts.append(attempt)
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**params)
            message = response.choices[0].message
            attempt.update(
                {
                    "response_model": response.model,
                    "response_id": response.id,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason,
                    "reasoning_content_chars": len(
                        getattr(message, "reasoning_content", None) or ""
                    ),
                }
            )
            return response
        except Exception as exc:
            # Avoid arbitrary exception messages containing headers or credentials.
            attempt["error_type"] = type(exc).__name__
            attempt["status_code"] = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                error = body.get("error", body)
                if isinstance(error, dict):
                    attempt["api_error"] = {
                        k: error[k] for k in ("code", "param", "message", "type") if k in error
                    }
            raise
        finally:
            attempt["seconds"] = time.perf_counter() - start


async def run(args):
    root = Path(__file__).resolve().parents[1]
    directory = (
        root / "runs" / ("settings-check-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"))
    )
    directory.mkdir(parents=True, exist_ok=False)
    source_root = root.parent / "src" / "giuriallm"
    report = {
        "live": args.live,
        "code_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in ("providers/openai_provider.py", "prompts/compiler.py")
        },
        "records": [],
    }
    destination = directory / "report.json"
    failures = 0
    for task in args.tasks:
        path = root / "configs" / f"{task}-panel.yaml"
        config = load_config(path)
        item = load_items(config.dataset, config.judge.reference.mode)[0]
        variant = variants_for(config, len(item.candidates))[0]
        compiled = PromptCompiler(config.judge, config.prompt).compile(item, variant)
        assert "justification" not in json.dumps(compiled.schema)
        assert not config.judge.outputs.explanation and not config.judge.outputs.feedback
        for model in config.panel.members:
            if args.models and model.id not in args.models:
                continue
            record = {
                "task": task,
                "model": model.id,
                "config_hash": config_hash(config),
                "dataset_sha256": hashlib.sha256(config.dataset.path.read_bytes()).hexdigest(),
                "item_id": item.id,
                "order": variant.order,
                "model_config": model.model_dump(mode="json"),
                "attempts": [],
            }
            report["records"].append(record)
            request = LLMRequest(
                model=model,
                system=compiled.system,
                user=compiled.user,
                schema=compiled.schema,
                tool_name=TOOL_NAME,
            )
            provider = OpenAIProvider()
            params = provider.build_params(request, strict=True)
            assert params["temperature"] == 0.0
            assert params.get("max_tokens", params.get("max_completion_tokens")) == 32768
            assert params["seed"] == 0
            if "deepseek" in model.id:
                assert params["extra_body"] == {"enable_thinking": False}
            else:
                assert params["reasoning_effort"] == "none"
            record["request"] = params
            record["status"] = "offline_pass"
            if args.live:
                from openai import AsyncOpenAI, DefaultAsyncHttpxClient

                env_var = model.options.get("api_key_env", "OPENAI_API_KEY")
                key = os.environ.get(env_var)
                if not key:
                    record["status"] = "missing_credential"
                    failures += 1
                else:

                    async def capture_request(http_request, record=record):
                        body = json.loads(http_request.content)
                        record.setdefault("wire_requests", []).append(
                            {
                                "url": str(http_request.url),
                                "parameters": {
                                    k: body[k]
                                    for k in (
                                        "model",
                                        "reasoning_effort",
                                        "enable_thinking",
                                        "temperature",
                                        "max_tokens",
                                        "max_completion_tokens",
                                        "seed",
                                    )
                                    if k in body
                                },
                            }
                        )

                    async with AsyncOpenAI(
                        api_key=key,
                        base_url=model.options.get("base_url"),
                        timeout=90,
                        max_retries=0,
                        http_client=DefaultAsyncHttpxClient(
                            event_hooks={"request": [capture_request]}
                        ),
                    ) as client:
                        provider = OpenAIProvider(
                            client=RecordingClient(client, record["attempts"])
                        )
                        try:
                            response = await provider.complete(request)
                            if not isinstance(response.payload, dict):
                                raise ValueError("missing structured verdict")
                            if set(response.payload) != set(compiled.schema["properties"]):
                                raise ValueError("unexpected or missing verdict fields")
                            for score in response.payload.get("scores", []):
                                if set(score) != {"criterion", "score"}:
                                    raise ValueError("unexpected or missing criterion fields")
                            verdict = parse_verdict(
                                response.payload, config.judge, variant, compiled.labels
                            )
                            record["payload"] = response.payload
                            record["parsed_verdict"] = verdict.model_dump(mode="json")
                            last = record["attempts"][-1]
                            usage = last.get("usage") or {}
                            details = usage.get("completion_tokens_details") or {}
                            reasoning = details.get("reasoning_tokens")
                            record["reasoning_tokens"] = reasoning
                            record["thinking_check"] = (
                                "not_reported" if reasoning is None else "zero_reported"
                            )
                            if last.get("reasoning_content_chars", 0) or (reasoning or 0) > 0:
                                record["thinking_check"] = "unexpected_reasoning"
                                raise ValueError(
                                    "provider returned reasoning despite disabled setting"
                                )
                            if response.stop_reason == "length":
                                raise ValueError("response hit output cap")
                            record["status"] = "live_pass"
                        except Exception as exc:
                            record["status"] = "failed"
                            record["error_type"] = type(exc).__name__
                            if isinstance(exc, ValueError):
                                record["validation_error"] = str(exc)
                            failures += 1
                        record["provider_notes"] = provider.notes
            destination.write_text(safe_json(report, indent=2, ensure_ascii=False) + "\n")
            print(f"{task:20} {model.id:34} {record['status']}", flush=True)
            if record["status"] == "failed":
                error = record["attempts"][-1].get("api_error", {}) if record["attempts"] else {}
                print(safe_json(error), flush=True)
    print(f"Report: {destination}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make paid API requests")
    parser.add_argument("--models", nargs="+", help="Only test these exact configured model IDs")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["conala-pointwise", "apr-pointwise", "qure-listwise", "cjb-testgen"],
    )
    raise SystemExit(asyncio.run(run(parser.parse_args())))
