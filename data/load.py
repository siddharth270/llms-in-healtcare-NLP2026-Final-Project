#!/usr/bin/env python3
"""The one way to load the dataset. Use this instead of reading the parquet.

Returns train/val/test as three separate frames, never one frame with a `split`
column -- a caller who forgets to filter that column gets silently wrong numbers
rather than an error. Row counts are checked against the split file, so a stale
parquet or splits built from a different seed fail loudly here.

    train, val, test = load_split("standard")
    train, val, test = load_split("heldout_theme", fold="hedging")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
PARQUET = DATA_DIR / "processed" / "meta_eval.parquet"
SPLIT_DIR = DATA_DIR / "splits"

THEMES = (
    "communication", "complex_responses", "context_seeking",
    "emergency_referrals", "global_health", "health_data_tasks", "hedging",
)


def load_dataset(path: Path = PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            "    python data/download.py\n"
            "    python data/build_dataset.py"
        )
    return pd.read_parquet(path)


def load_split(
    name: str = "standard",
    fold: str | None = None,
    path: Path = PARQUET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) for a split definition, verified against its counts."""
    spec_path = SPLIT_DIR / f"{name}.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found. Run python data/build_splits.py")
    spec = json.loads(spec_path.read_text())

    splits, counts = spec["splits"], spec["counts"]
    if fold is not None:
        if fold not in splits:
            raise KeyError(f"fold {fold!r} not in {name}; options: {sorted(splits)}")
        splits, counts = splits[fold], counts[fold]
    elif "train" not in splits:
        raise ValueError(f"{name} is a multi-fold split; pass fold=, one of {sorted(splits)}")

    df = load_dataset(path)
    out = []
    for part in ("train", "val", "test"):
        ids = set(splits[part])
        frame = df[df.prompt_id.isin(ids)].reset_index(drop=True)
        expected = counts[part]
        if len(frame) != expected["rows"]:
            raise ValueError(
                f"{name}/{part}: parquet has {len(frame):,} rows but the split file "
                f"expects {expected['rows']:,}. The parquet is stale or the splits "
                f"were rebuilt -- regenerate both."
            )
        out.append(frame)

    train, val, test = out
    assert not (set(train.prompt_id) & set(test.prompt_id)), "train/test leakage"
    return train, val, test


def load_heldout_folds(path: Path = PARQUET):
    """Yield (theme, train, val, test) for each leave-one-theme-out fold."""
    for theme in THEMES:
        yield (theme, *load_split("heldout_theme", fold=theme, path=path))
