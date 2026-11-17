from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SIMPLE_EVALS = PROJECT_ROOT.parent / "simple-evals"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
PROMPT_KINDS = ("plain", "cot", "fewshot")
TEST_ROWS = 4438

VERDICT_STEM = '"criteria_met":'
DIRECT_PREFILL = '```json\n{\n  "explanation": "N/A",\n  ' + VERDICT_STEM


def load_grader_template(repo_path: Path) -> str:
    module_path = repo_path / "healthbench_eval.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Missing {module_path}; clone openai/simple-evals next to this project "
            "or pass --simple-evals-path."
        )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if not any(isinstance(t, ast.Name) and t.id == "GRADER_TEMPLATE" for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            template = value.value
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "strip"
            and isinstance(value.func.value, ast.Constant)
            and isinstance(value.func.value.value, str)
        ):
            template = value.func.value.value.strip()
        else:
            break
        if "<<conversation>>" not in template or "<<rubric_item>>" not in template:
            raise ValueError("GRADER_TEMPLATE is missing its placeholders.")
        return template
    raise ValueError(f"Could not read GRADER_TEMPLATE from {module_path}.")


def load_split_frame(
    split: str,
    part: str,
    limit: int | None = None,
    sample: int | None = None,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    split_path = root / "data" / "splits" / f"{split}.json"
    data_path = root / "data" / "processed" / "meta_eval.parquet"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run: python data/download.py && "
            "python data/build_dataset.py"
        )
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    prompt_ids = split_data.get("splits", {}).get(part)
    if not isinstance(prompt_ids, list):
        raise ValueError(f"Split {split!r} has no {part!r} prompt ID list.")
    frame = pd.read_parquet(data_path)
    frame = frame.loc[frame["prompt_id"].isin(set(prompt_ids))].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No parquet rows matched {split}/{part} prompt IDs.")
    expected = split_data.get("counts", {}).get(part, {}).get("rows")
    if expected is not None and len(frame) != expected:
        raise ValueError(
            f"{split}/{part}: parquet has {len(frame)} rows but the split file expects "
            f"{expected}. The parquet or the splits are stale."
        )
    if limit is not None and sample is not None:
        raise ValueError("Use either --limit or --sample, not both.")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        frame = frame.iloc[:limit].reset_index(drop=True)
    if sample is not None:
        if sample < 1:
            raise ValueError("--sample must be positive.")
        frame = frame.sample(n=min(sample, len(frame)), random_state=0).sort_index()
        frame = frame.reset_index(drop=True)
    required = {
        "completion_id", "conversation", "completion", "criterion",
        "binary_labels", "theme", "axis", "unanimous", "hard_label",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Parquet is missing columns: {', '.join(sorted(missing))}")
    return frame


def make_user_prompt(template: str, row: Any) -> str:
    conversation = f"{row.conversation}\n\nassistant: {row.completion}"
    return template.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", str(row.criterion)
    )


def shot_answer(met: bool) -> str:
    return (
        '```json\n{\n  "explanation": "N/A",\n  "criteria_met": '
        + ("true" if met else "false")
        + "\n}\n```"
    )


def select_shots(train: pd.DataFrame, k: int, seed: int, max_chars: int) -> pd.DataFrame:
    if k < 2:
        raise ValueError("--shots must be at least 2 (one met, one not met).")
    size = train["conversation"].str.len() + train["completion"].str.len() + train["criterion"].str.len()
    pool = train.loc[train["unanimous"] & (size <= max_chars)]
    pool = pool.drop_duplicates("prompt_id")
    n_met, n_not = (k + 1) // 2, k // 2
    met, not_met = pool.loc[pool["hard_label"]], pool.loc[~pool["hard_label"]]
    if len(met) < n_met or len(not_met) < n_not:
        raise ValueError("Not enough short unanimous train examples; raise --shots-max-chars.")
    picked = pd.concat([met.sample(n_met, random_state=seed), not_met.sample(n_not, random_state=seed)])
    return picked.sample(frac=1, random_state=seed)


def build_messages(template: str, row: Any, kind: str, shots: pd.DataFrame | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if kind == "fewshot":
        if shots is None or shots.empty:
            raise ValueError("fewshot needs examples.")
        for shot in shots.itertuples():
            messages.append({"role": "user", "content": make_user_prompt(template, shot)})
            messages.append({"role": "assistant", "content": shot_answer(bool(shot.hard_label))})
    messages.append({"role": "user", "content": make_user_prompt(template, row)})
    return messages


def verdict_prefix_from_generation(gen: str) -> tuple[str, bool, bool]:
    text = gen.lstrip()
    i = text.find(VERDICT_STEM)
    if i >= 0:
        return text[:i] + VERDICT_STEM, False, False
    if '"explanation"' in text:
        return text.rstrip() + '",\n  ' + VERDICT_STEM, True, False
    return DIRECT_PREFILL, True, True


def judge_chunk(rows: list[Any], scorer: Any, kind: str, template: str,
                shots: pd.DataFrame | None, max_new_tokens: int) -> list[dict[str, Any]]:
    messages = [build_messages(template, row, kind, shots) for row in rows]
    gens: list[str | None] = [None] * len(rows)
    truncated = [False] * len(rows)
    fallback = [False] * len(rows)
    if kind == "cot":
        gens = scorer.generate([scorer.render(m, "") for m in messages], max_new_tokens)
        prefixes = []
        for i, gen in enumerate(gens):
            prefix, truncated[i], fallback[i] = verdict_prefix_from_generation(gen)
            prefixes.append(prefix)
    else:
        prefixes = [DIRECT_PREFILL] * len(rows)
    scores = scorer.score([scorer.render(m, p) for m, p in zip(messages, prefixes)])
    return [
        {"p_met": float(p), "mass": float(mass), "gen": gens[i],
         "truncated": truncated[i], "fallback": fallback[i]}
        for i, (p, mass) in enumerate(scores)
    ]


class HFScorer:
    def __init__(self, model_id: str, dtype: str, quantize: str, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.warned_logits = False
        self.batch_size = max(1, batch_size)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        if dtype == "auto":
            if torch.cuda.is_available():
                native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
                chosen = torch.bfloat16 if native_bf16 else torch.float16
            else:
                chosen = torch.float32
        else:
            chosen = getattr(torch, dtype)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if quantize != "none":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=chosen,
                                   bnb_4bit_quant_type="nf4")
                if quantize == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            )
        print(f"Loading {model_id} (dtype={chosen}, quantize={quantize}) ...", flush=True)
        self.model = self._load(model_id, AutoModelForCausalLM, kwargs, chosen)
        self.model.eval()

        stem = self.tok(VERDICT_STEM, add_special_tokens=False)["input_ids"]

        def first_new_token(word: str) -> int:
            full = self.tok(f"{VERDICT_STEM} {word}", add_special_tokens=False)["input_ids"]
            if full[: len(stem)] != stem or len(full) <= len(stem):
                raise RuntimeError("Unexpected tokenization of the verdict stem.")
            return full[len(stem)]

        self.true_id, self.false_id = first_new_token("true"), first_new_token("false")
        if self.true_id == self.false_id:
            raise RuntimeError("'true' and 'false' map to the same token.")

    @staticmethod
    def _load(model_id, auto_cls, kwargs, dtype):
        attempts = ({"dtype": dtype}, {"torch_dtype": dtype})
        last: Exception | None = None
        for extra in attempts:
            try:
                return auto_cls.from_pretrained(model_id, **kwargs, **extra)
            except ValueError as exc:
                last = exc
                from transformers import AutoModelForImageTextToText
                try:
                    return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs, **extra)
                except TypeError as exc2:
                    last = exc2
            except TypeError as exc:
                last = exc
        raise RuntimeError(f"Could not load {model_id}: {last}")

    def render(self, messages: list[dict[str, str]], prefill: str) -> str:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return text + prefill

    def _batches(self, prompts: list[str]):
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            yield idx, [prompts[i] for i in idx]

    def _with_oom_backoff(self, fn, batch: list[str]):
        try:
            return fn(batch)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._with_oom_backoff(fn, batch[:mid]) + self._with_oom_backoff(fn, batch[mid:])

    def _score_batch(self, batch: list[str]) -> list[tuple[float, float]]:
        torch = self.torch
        enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(self.model.device)
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            try:
                out = self.model(**enc, position_ids=position_ids, use_cache=False, logits_to_keep=1)
            except TypeError:
                out = self.model(**enc, position_ids=position_ids, use_cache=False)
        if out.logits.shape[1] != 1 and not self.warned_logits:
            print("Warning: the model returned logits for every position; memory use will be high.", flush=True)
            self.warned_logits = True
        logits = out.logits[:, -1, :].float()
        pair = logits[:, [self.true_id, self.false_id]]
        p_met = torch.sigmoid(pair[:, 0] - pair[:, 1])
        mass = torch.exp(torch.logsumexp(pair, dim=-1) - torch.logsumexp(logits, dim=-1))
        return list(zip(p_met.tolist(), mass.tolist()))

    def score(self, prompts: list[str]) -> list[tuple[float, float]]:
        results: list[Any] = [None] * len(prompts)
        for idx, batch in self._batches(prompts):
            for i, r in zip(idx, self._with_oom_backoff(self._score_batch, batch)):
                results[i] = r
        return results

    def _generate_batch(self, batch: list[str], max_new_tokens: int) -> list[str]:
        torch = self.torch
        enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        results: list[Any] = [None] * len(prompts)
        for idx, batch in self._batches(prompts):
            outs = self._with_oom_backoff(lambda b: self._generate_batch(b, max_new_tokens), batch)
            for i, r in zip(idx, outs):
                results[i] = r
        return results

    def peak_gb(self) -> float | None:
        if self.torch.cuda.is_available():
            return self.torch.cuda.max_memory_allocated() / 1e9
        return None


def model_tag(model_id: str) -> str:
    return re.sub(r"[^a-z0-9.\-]+", "-", model_id.split("/")[-1].lower()).strip("-")


def write_standard_predictions(path: Path, frame: pd.DataFrame, p_met: list[float]) -> None:
    preds = pd.DataFrame({
        "completion_id": frame["completion_id"].astype(str).to_numpy(),
        "criterion": frame["criterion"].astype(str).to_numpy(),
        "p_met": pd.Series(p_met, dtype="float64").to_numpy(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    checks = (
        (f"len == {TEST_ROWS}", len(saved) == TEST_ROWS),
        ("p_met is within [0, 1] and has no NaN", bool(saved["p_met"].between(0.0, 1.0).all())),
        ("no duplicate (completion_id, criterion) pairs",
         not saved.duplicated(["completion_id", "criterion"]).any()),
        ("completion_id and criterion are str",
         all(saved[c].map(lambda v: isinstance(v, str)).all() for c in ("completion_id", "criterion"))),
        ("p_met has more than 2 distinct values (not thresholded)", saved["p_met"].nunique() > 2),
    )
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    if not all(ok for _, ok in checks):
        raise RuntimeError("Standard prediction parquet validation failed.")


def config_signature(cfg: dict[str, Any]) -> str:
    keys = ("model", "prompt", "split", "part", "limit", "sample", "shots",
            "max_new_tokens", "dtype", "quantize", "template_sha")
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, ensure_ascii=False)


def read_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[int(rec["id"])] = rec
    return done


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open-weight LLM judges for HealthBench.")
    p.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-1.7B")
    p.add_argument("--prompt", choices=PROMPT_KINDS, default="plain")
    p.add_argument("--split", default="standard")
    p.add_argument("--part", choices=("val", "test"), default="val")
    p.add_argument("--limit", type=int, default=None, help="first N rows (debugging)")
    p.add_argument("--sample", type=int, default=None, help="random N rows, fixed seed (val slices)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=320, help="cot only")
    p.add_argument("--shots", type=int, default=4, help="fewshot only")
    p.add_argument("--shots-max-chars", type=int, default=2500)
    p.add_argument("--seed", type=int, default=20261111, help="fewshot example selection")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--quantize", choices=("none", "4bit", "8bit"), default="none")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--simple-evals-path", type=Path, default=DEFAULT_SIMPLE_EVALS)
    p.add_argument("--dry-run", action="store_true", help="print one prompt and exit; loads no model")
    return p


def run(args: argparse.Namespace, scorer: Any = None, data_root: Path = PROJECT_ROOT) -> int:
    template = load_grader_template(args.simple_evals_path)
    template_sha = hashlib.sha256(template.encode()).hexdigest()[:12]
    frame = load_split_frame(args.split, args.part, args.limit, args.sample, root=data_root)

    shots = None
    shot_info: list[dict[str, Any]] = []
    if args.prompt == "fewshot":
        train = load_split_frame(args.split, "train", root=data_root)
        shots = select_shots(train, args.shots, args.seed, args.shots_max_chars)
        shot_info = [{"completion_id": s.completion_id, "criterion": str(s.criterion)[:80],
                      "met": bool(s.hard_label)} for s in shots.itertuples()]

    if args.dry_run:
        msgs = build_messages(template, next(frame.itertuples()), args.prompt, shots)
        for m in msgs:
            print(f"----- {m['role']} -----\n{m['content']}\n")
        print(f"----- assistant prefill ({'none; the model writes this' if args.prompt == 'cot' else 'verdict read after this'}) -----")
        print("" if args.prompt == "cot" else DIRECT_PREFILL)
        print(f"\nRows in this run: {len(frame)} | template sha: {template_sha}")
        return 0

    tag = model_tag(args.model)
    suffix = f"_s{args.sample}" if args.sample else (f"_n{args.limit}" if args.limit else "")
    run_name = args.run_name or f"{tag}_{args.prompt}_{args.part}{suffix}"
    run_dir = args.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model": args.model, "prompt": args.prompt, "split": args.split, "part": args.part,
        "limit": args.limit, "sample": args.sample, "shots": shot_info,
        "max_new_tokens": args.max_new_tokens if args.prompt == "cot" else None,
        "dtype": args.dtype, "quantize": args.quantize, "template_sha": template_sha,
        "rows": len(frame), "started": datetime.now().isoformat(timespec="seconds"),
    }
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if config_signature(old) != config_signature(cfg):
            raise ValueError(f"{run_dir} holds a run with different settings; pass a new --run-name.")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_path = run_dir / "rows.jsonl"
    done = read_done(rows_path)
    todo = [i for i in range(len(frame)) if i not in done]
    print(f"{run_name}: {len(done)} rows already done, {len(todo)} to go.", flush=True)

    if todo:
        if scorer is None:
            scorer = HFScorer(args.model, args.dtype, args.quantize, args.batch_size)
        chunk = max(args.batch_size * 4, 16)
        started = time.perf_counter()
        with rows_path.open("a", encoding="utf-8") as sink:
            for s in range(0, len(todo), chunk):
                ids = todo[s : s + chunk]
                rows = [frame.iloc[i] for i in ids]
                t0 = time.perf_counter()
                recs = judge_chunk([r for r in rows], scorer, args.prompt, template, shots, args.max_new_tokens)
                per_row = (time.perf_counter() - t0) / len(ids)
                for i, rec in zip(ids, recs):
                    if rec["p_met"] != rec["p_met"]:
                        raise RuntimeError(
                            "Got NaN probabilities. Try --dtype float32 or --dtype bfloat16."
                        )
                    rec.update({"id": i, "sec": per_row})
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                n_done = s + len(ids)
                elapsed = time.perf_counter() - started
                eta = elapsed / n_done * (len(todo) - n_done)
                print(f"  {n_done}/{len(todo)} rows | {elapsed/60:.1f} min elapsed | ~{eta/60:.1f} min left", flush=True)
        done = read_done(rows_path)

    p_met = [done[i]["p_met"] for i in range(len(frame))]
    pd.DataFrame({
        "id": range(len(frame)), "completion_id": frame["completion_id"],
        "p_met": p_met, "pred": ["met" if p >= 0.5 else "not_met" for p in p_met],
    }).to_csv(run_dir / "predictions.csv", index=False)

    n = len(frame)
    secs = sum(done[i]["sec"] for i in range(n))
    mass = sum(done[i]["mass"] for i in range(n)) / n
    print(f"\nRows: {n} | mean p_met: {sum(p_met)/n:.3f} | share predicted met: {sum(p >= 0.5 for p in p_met)/n:.3f}")
    print(f"Probability mass on true/false tokens (should be near 1): {mass:.3f}")
    if args.prompt == "cot":
        print(f"Reasoning cut off before verdict: {sum(done[i]['truncated'] for i in range(n))}/{n} "
              f"| fell back to direct: {sum(done[i]['fallback'] for i in range(n))}/{n}")
    print(f"Compute time: {secs/60:.1f} min total | {secs*1000/n:.1f} s per 1,000 judgments")
    peak = scorer.peak_gb() if scorer is not None and hasattr(scorer, "peak_gb") else None
    if peak:
        print(f"Peak GPU memory this session: {peak:.1f} GB")

    if args.part == "test" and args.limit is None and args.sample is None:
        write_standard_predictions(args.results_dir / f"preds_{tag}_{args.prompt}.parquet", frame, p_met)
    else:
        print("Standard prediction parquet skipped: needs the full test split (no --limit/--sample).")

    try:
        from eval.metrics import evaluate, format_report
        print(format_report(evaluate(frame, p_met)))
    except ImportError as exc:
        print(f"(metrics report skipped: {exc})")
    print(f"Run folder: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
