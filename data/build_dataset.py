#!/usr/bin/env python3
"""Flatten meta_eval.jsonl into data/processed/meta_eval.parquet.

One row per (conversation, response, criterion) with physician votes and the
derived label columns. Themes and axes are joined in from eval.jsonl, which is
the only place they exist -- meta_eval.jsonl carries just a flat `category`.

The output schema is the contract both model owners code against; see
data/README.md for the input rules that go with it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_PATH = OUT_DIR / "meta_eval.parquet"


def render(messages: list[dict]) -> str:
    """The conversation as plain text. Models tokenize this themselves."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_lookups() -> tuple[dict[str, str], dict[str, str]]:
    """theme per prompt_id, and axis per cluster, both from eval.jsonl."""
    themes: dict[str, str] = {}
    axes: dict[str, set[str]] = collections.defaultdict(set)

    with (RAW_DIR / "eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            tag = next((t for t in r["example_tags"] if t.startswith("theme:")), None)
            if tag:
                themes[r["prompt_id"]] = tag.split(":", 1)[1]

            # cluster-level rubric items carry both a cluster: and an axis: tag,
            # which is the only link from a meta_eval category to its axis
            for item in r["rubrics"]:
                clusters = [t for t in item["tags"] if t.startswith("cluster:")]
                axis = next((t for t in item["tags"] if t.startswith("axis:")), None)
                if clusters and axis:
                    for c in clusters:
                        axes[c].add(axis.split(":", 1)[1])

    multi = {c: a for c, a in axes.items() if len(a) > 1}
    assert not multi, f"clusters mapping to >1 axis: {multi}"
    return themes, {c: a.pop() for c, a in axes.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    themes, axes = build_lookups()
    print(f"lookups: {len(themes):,} prompt_id->theme, {len(axes)} cluster->axis")

    records = []
    with (RAW_DIR / "meta_eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)  # `canary` is simply never read
            labels = r["binary_labels"]
            n_met = sum(labels)
            soft = n_met / len(labels)
            records.append(
                {
                    "prompt_id": r["prompt_id"],
                    "completion_id": r["completion_id"],
                    "conversation": render(r["prompt"]),
                    "completion": r["completion"],
                    "criterion": r["rubric"],
                    "cluster": r["category"],
                    "theme": themes[r["prompt_id"]],
                    "axis": axes[r["category"]],
                    "binary_labels": labels,
                    "physician_ids": r["anonymized_physician_ids"],
                    "n_physicians": len(labels),
                    "n_met": n_met,
                    "soft_label": soft,
                    # explicit comparison: round(0.5) is 0 in Python, which
                    # would silently send all 6,020 tied rows to not-met
                    "hard_label": soft >= 0.5,
                    "unanimous": soft in (0.0, 1.0),
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["prompt_id", "completion_id", "criterion"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n{len(df):,} rows x {len(df.columns)} columns -> "
          f"{args.out.relative_to(DATA_DIR.parent)} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"\nconversations {df.prompt_id.nunique():,}  "
          f"completions {df.completion_id.nunique():,}  "
          f"criteria {df.criterion.nunique()}  "
          f"clusters {df.cluster.nunique()}")
    print(f"met {100 * df.hard_label.mean():.1f}%  "
          f"unanimous {100 * df.unanimous.mean():.1f}%  "
          f"physician votes {df.n_physicians.sum():,}")
    print("\nrows per axis:")
    print(df.axis.value_counts().to_string())
    print("\nrows per theme:")
    print(df.theme.value_counts().to_string())


if __name__ == "__main__":
    main()
