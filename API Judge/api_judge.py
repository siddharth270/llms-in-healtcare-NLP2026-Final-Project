from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_SIMPLE_EVALS = PROJECT_ROOT.parent / "simple-evals"
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
THINKING_MODEL_MARKERS = ("reasoner", "reasoning", "thinking", "deepseek-r1", "-r1")


@dataclass(frozen=True)
class CallResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    dollar_cost: float
    cache_hit: bool
    cache_key: str


@dataclass(frozen=True)
class Judgment:
    row_id: int
    p_met: float
    parse_failed: bool
    calls: tuple[dict[str, Any], ...]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Config file must contain a JSON object.")
    return value


def configured_value(args: argparse.Namespace, config: dict[str, Any], name: str):
    value = getattr(args, name)
    return value if value is not None else config.get(name)


def load_healthbench_helpers(repo_path: Path):
    module_path = repo_path / "healthbench_eval.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Missing {module_path}; clone openai/simple-evals outside this project "
            "or pass --simple-evals-path."
        )

    package_name = f"_simple_evals_external_{hashlib.sha1(str(repo_path).encode()).hexdigest()[:10]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_path)]
    sys.modules[package_name] = package
    qualified_name = f"{package_name}.healthbench_eval"
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    if not isinstance(module.GRADER_TEMPLATE, str) or not callable(module.parse_json_to_dict):
        raise TypeError("Upstream HealthBench template/parser have an unexpected shape.")
    return module.GRADER_TEMPLATE, module.parse_json_to_dict


def cache_key(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    encoded = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                dollar_cost REAL NOT NULL
            )
            """
        )
        self.connection.commit()
        self.lock = threading.Lock()

    def get(self, key: str) -> CallResult | None:
        with self.lock:
            row = self.connection.execute(
                """
                SELECT content, prompt_tokens, completion_tokens, latency_ms, dollar_cost
                FROM responses WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CallResult(
            content=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            latency_ms=row[3],
            dollar_cost=0.0,
            cache_hit=True,
            cache_key=key,
        )

    def put(self, result: CallResult) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO responses
                (cache_key, content, prompt_tokens, completion_tokens, latency_ms, dollar_cost)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.cache_key,
                    result.content,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.latency_ms,
                    result.dollar_cost,
                ),
            )
            self.connection.commit()

    def close(self) -> None:
        with self.lock:
            self.connection.close()


def get_retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(60.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, (2**attempt) + random.uniform(0.0, 0.5))


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES


def create_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Set it in the environment or project .env file."
        )
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, max_retries=0, timeout=90.0)


def list_models(client: OpenAI, max_retries: int) -> list[str]:
    print("Requesting model metadata only; no judgment tokens are used (estimated cost: $0).")
    for attempt in range(max_retries + 1):
        try:
            response = client.models.list()
            break
        except Exception as exc:
            if not is_retryable(exc) or attempt >= max_retries:
                raise
            time.sleep(get_retry_delay(exc, attempt))
    else:
        raise RuntimeError("Model-list retry loop exited unexpectedly.")
    model_ids = sorted({item.id for item in response.data})
    print("Models available to this key:")
    for model_id in model_ids:
        print(f"  {model_id}")
    return model_ids


def choose_non_thinking_model(model_ids: list[str], requested: str | None) -> str | None:
    eligible = [
        model_id for model_id in model_ids
        if not any(marker in model_id.lower() for marker in THINKING_MODEL_MARKERS)
    ]
    if requested:
        if requested not in model_ids:
            raise ValueError(f"Configured model {requested!r} is not in the model list.")
        if requested not in eligible:
            raise ValueError(f"Configured model {requested!r} appears to be a thinking model.")
        return requested
    preferred = ("deepseek-flash", "deepseek-chat", "deepseek-v3", "deepseek-v3.2")
    return next((model for model in preferred if model in eligible), None)


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, math.ceil(len(prompt.encode("utf-8")) / 3))


def estimate_cost(
    prompts: list[str],
    max_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> tuple[int, int, float]:
    input_tokens = sum(estimate_prompt_tokens(prompt) for prompt in prompts)
    output_token_cap = len(prompts) * max_tokens
    total = (
        input_tokens * input_price_per_1m + output_token_cap * output_price_per_1m
    ) / 1_000_000
    return input_tokens, output_token_cap, total


def parse_criteria_met(content: str, parse_json_to_dict) -> bool | None:
    parsed = parse_json_to_dict(content)
    if isinstance(parsed, dict) and isinstance(parsed.get("criteria_met"), bool):
        return parsed["criteria_met"]
    match = re.search(
        r"""["']?criteria_met["']?\s*:\s*(true|false)\b""",
        content,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower() == "true"
    return None


def load_split_frame(split: str, part: str, limit: int | None) -> pd.DataFrame:
    split_path = PROJECT_ROOT / "data" / "splits" / f"{split}.json"
    data_path = PROJECT_ROOT / "data" / "processed" / "meta_eval.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Processed parquet does not exist: {data_path}")
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    prompt_ids = split_data.get("splits", {}).get(part)
    if not isinstance(prompt_ids, list):
        raise ValueError(f"Split {split!r} has no {part!r} prompt ID list.")
    frame = pd.read_parquet(data_path)
    if "prompt_id" not in frame.columns:
        raise ValueError("Parquet is missing required prompt_id column.")
    selected_ids = set(prompt_ids)
    frame = frame.loc[frame["prompt_id"].isin(selected_ids)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No parquet rows matched {split}/{part} prompt IDs.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        frame = frame.iloc[:limit].copy().reset_index(drop=True)
    required = {"conversation", "completion", "criterion", "binary_labels", "theme", "axis", "unanimous"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {', '.join(sorted(missing))}")
    return frame


def make_prompt(template: str, row: Any) -> str:
    conversation = f"{row.conversation}\n\nassistant: {row.completion}"
    return template.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", str(row.criterion)
    )


def resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    names = (
        "model",
        "temperature",
        "max_tokens",
        "input_price_per_1m",
        "output_price_per_1m",
    )
    settings = {name: configured_value(args, config, name) for name in names}
    missing = [name for name, value in settings.items() if value is None]
    if missing:
        raise ValueError(
            "Missing settings (supply CLI flags or --config): "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise ValueError("Model name must be a non-empty string.")
    if float(settings["temperature"]) != 0.0:
        raise ValueError("This judge requires temperature 0.")
    settings["temperature"] = 0.0
    settings["max_tokens"] = int(settings["max_tokens"])
    if settings["max_tokens"] < 1:
        raise ValueError("max_tokens must be positive.")
    for name in ("input_price_per_1m", "output_price_per_1m"):
        settings[name] = float(settings[name])
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
    return settings


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_judgment(
    row_id: int,
    prompt: str,
    *,
    client: OpenAI,
    cache: ResponseCache,
    model: str,
    temperature: float,
    max_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
    parse_json_to_dict,
    max_retries: int,
) -> Judgment:
    call_logs: list[dict[str, Any]] = []

    def call(prompt_text: str, call_type: str) -> str:
        key = cache_key(model, prompt_text, temperature, max_tokens)
        cached = cache.get(key)
        if cached is not None:
            call_logs.append(
                {
                    "id": row_id,
                    "call_type": call_type,
                    "prompt_tokens": cached.prompt_tokens,
                    "completion_tokens": cached.completion_tokens,
                    "latency_ms": 0.0,
                    "dollar_cost": 0.0,
                    "cache_hit": True,
                    "cache_key": key,
                    "status": "cached",
                }
            )
            return cached.content

        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                latency_ms = (time.perf_counter() - started) * 1000
                content = response.choices[0].message.content or ""
                usage = response.usage
                if usage is None:
                    raise RuntimeError("DeepSeek response omitted token-usage data.")
                prompt_tokens = int(usage.prompt_tokens)
                completion_tokens = int(usage.completion_tokens)
                cost = (
                    prompt_tokens * input_price_per_1m
                    + completion_tokens * output_price_per_1m
                ) / 1_000_000
                result = CallResult(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    dollar_cost=cost,
                    cache_hit=False,
                    cache_key=key,
                )
                cache.put(result)
                call_logs.append(
                    {
                        "id": row_id,
                        "call_type": call_type,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_ms": round(latency_ms, 3),
                        "dollar_cost": cost,
                        "cache_hit": False,
                        "cache_key": key,
                        "status": "ok",
                    }
                )
                return content
            except Exception as exc:
                if not is_retryable(exc) or attempt >= max_retries:
                    raise
                time.sleep(get_retry_delay(exc, attempt))
        raise RuntimeError("API retry loop exited unexpectedly.")

    result = call(prompt, "primary")
    verdict = parse_criteria_met(result, parse_json_to_dict)
    if verdict is None:
        retry_prompt = (
            prompt
            + "\n\nYour previous answer could not be parsed. Return only a valid JSON "
            'object with a boolean "criteria_met" field and a string "explanation" field.'
        )
        result = call(retry_prompt, "parse_retry")
        verdict = parse_criteria_met(result, parse_json_to_dict)
    return Judgment(
        row_id=row_id,
        p_met=1.0 if verdict is None or verdict else 0.0,
        parse_failed=verdict is None,
        calls=tuple(call_logs),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="standard")
    parser.add_argument("--part", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--input-price-per-1m", type=float)
    parser.add_argument("--output-price-per-1m", type=float)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--simple-evals-path", type=Path, default=DEFAULT_SIMPLE_EVALS)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available to the key and recommend a non-thinking model, then exit.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print a rough, no-API pilot cost estimate and exit.",
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Required to proceed with API judgments after displaying the estimate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        load_env_file(args.env_file)
        if args.workers < 1 or args.workers > 8:
            raise ValueError("--workers must be between 1 and 8.")
        if args.max_retries < 0:
            raise ValueError("--max-retries cannot be negative.")

        if args.list_models:
            client = create_client()
            model_ids = list_models(client, args.max_retries)
            selected = choose_non_thinking_model(model_ids, args.model)
            if selected:
                print(f"Recommended non-thinking chat model: {selected}")
            else:
                print("No recognized non-thinking chat model was found; choose one explicitly.")
            return 0

        settings = resolved_settings(args)
        frame = load_split_frame(args.split, args.part, args.limit)
        template, parse_json_to_dict = load_healthbench_helpers(args.simple_evals_path)
        prompts = [make_prompt(template, row) for row in frame.itertuples(index=False)]
        input_estimate, output_cap, estimated_cost = estimate_cost(
            prompts,
            settings["max_tokens"],
            settings["input_price_per_1m"],
            settings["output_price_per_1m"],
        )
        print(
            f"Estimated {len(prompts)}-judgment cost (rough input-token estimate; "
            f"completion upper cap): ${estimated_cost:.6f} "
            f"({input_estimate:,} estimated input tokens + {output_cap:,} max output tokens)."
        )
        if args.estimate_only:
            return 0
        if not args.confirm_cost:
            print("No API requests made. Review the estimate, then rerun with --confirm-cost.")
            return 2

        client = create_client()
        model_ids = list_models(client, args.max_retries)
        selected_model = choose_non_thinking_model(model_ids, settings["model"])
        if selected_model is None:
            raise ValueError("No non-thinking model is available for this key.")
        print(f"Selected non-thinking chat model: {selected_model}")

        run_name = args.run_name or (
            f"{args.split}_{args.part}_{selected_model}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        if Path(run_name).name != run_name or run_name in ("", ".", ".."):
            raise ValueError("--run-name must be a single path component.")
        results_root = PROJECT_ROOT / "results"
        run_dir = results_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cache = ResponseCache(results_root / "cache" / "api_judge.sqlite3")
        log_lock = threading.Lock()
        call_logs: list[dict[str, Any]] = []
        judgments: list[Judgment] = []
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_judgment,
                        row_id,
                        prompt,
                        client=client,
                        cache=cache,
                        model=selected_model,
                        temperature=settings["temperature"],
                        max_tokens=settings["max_tokens"],
                        input_price_per_1m=settings["input_price_per_1m"],
                        output_price_per_1m=settings["output_price_per_1m"],
                        parse_json_to_dict=parse_json_to_dict,
                        max_retries=args.max_retries,
                    ): row_id
                    for row_id, prompt in enumerate(prompts)
                }
                for future in as_completed(futures):
                    judgment = future.result()
                    judgments.append(judgment)
                    with log_lock:
                        call_logs.extend(judgment.calls)
            judgments.sort(key=lambda result: result.row_id)
        finally:
            cache.close()

        write_csv(
            run_dir / "predictions.csv",
            ["id", "p_met", "pred"],
            [
                {
                    "id": result.row_id,
                    "p_met": result.p_met,
                    "pred": "met" if result.p_met == 1.0 else "not_met",
                }
                for result in judgments
            ],
        )
        write_csv(
            run_dir / "cost_log.csv",
            [
                "id",
                "call_type",
                "prompt_tokens",
                "completion_tokens",
                "latency_ms",
                "dollar_cost",
                "cache_hit",
                "cache_key",
                "status",
            ],
            call_logs,
        )

        total_cost = sum(row["dollar_cost"] for row in call_logs)
        per_1k = total_cost * 1000 / len(judgments) if judgments else 0.0
        parse_failures = sum(result.parse_failed for result in judgments)
        print(f"Parse failures: {parse_failures}/{len(judgments)} ({parse_failures / len(judgments):.2%})")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Cost per 1,000 judgments: ${per_1k:.6f}")

        from eval.metrics import evaluate, format_report

        metrics = evaluate(frame, [result.p_met for result in judgments])
        print(format_report(metrics))
        print(f"Predictions: {run_dir / 'predictions.csv'}")
        print(f"Cost log: {run_dir / 'cost_log.csv'}")
        return 0
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
