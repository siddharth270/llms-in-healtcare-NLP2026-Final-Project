#!/usr/bin/env python3
"""Build the train/val/test splits described in data/SPLIT_DESIGN.md.

Splits are defined over conversations (prompt_id), never over rows: each
conversation yields ~8 rows (4 responses x ~2 criteria) and responses to the
same prompt are often near-identical, so a row-level split would put
paraphrases of training items into the test set.

Writes committed ID lists to data/splits/. Do not regenerate these once anyone
has started training -- re-rolling silently invalidates every comparison made
before the change.
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import date
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
MANIFEST = DATA_DIR / "manifest.json"

SEED = 20261111
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def load_conversations() -> dict[str, dict]:
    """One entry per prompt_id, carrying its stratum, theme and label stats."""
    clusters = collections.defaultdict(set)
    n_rows = collections.Counter()
    n_met = collections.Counter()
    n_disagree = collections.Counter()

    with (RAW_DIR / "meta_eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            pid = r["prompt_id"]
            clusters[pid].add(r["category"])
            n_rows[pid] += 1
            labels = r["binary_labels"]
            soft = sum(labels) / len(labels)
            if soft >= 0.5:
                n_met[pid] += 1
            if 0 < soft < 1:
                n_disagree[pid] += 1

    # theme comes from eval.jsonl's example_tags -- parsing it out of the
    # cluster name is fragile because several themes contain underscores.
    themes = {}
    with (RAW_DIR / "eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            tag = next(
                (t for t in r["example_tags"] if t.startswith("theme:")), None
            )
            if tag:
                themes[r["prompt_id"]] = tag.split(":", 1)[1]

    return {
        pid: {
            "cluster_set": "|".join(sorted(cs)),  # sorted: {A,B} and {B,A} are one stratum
            "theme": themes[pid],
            "n_rows": n_rows[pid],
            "n_met": n_met[pid],
            "n_disagree": n_disagree[pid],
        }
        for pid, cs in clusters.items()
    }


def stratified_split(pids, convs, frac, seed):
    """Split `pids` into (majority, minority) with `frac` in the minority."""
    strata = [convs[p]["cluster_set"] for p in pids]
    return train_test_split(
        pids, test_size=frac, stratify=strata, random_state=seed
    )


def build_standard(convs: dict, seed: int) -> dict:
    pids = sorted(convs)  # sort first so the seed alone determines the result
    rest, test = stratified_split(pids, convs, TEST_FRAC, seed)
    # val is VAL_FRAC of the ORIGINAL total, so rescale against what's left
    train, val = stratified_split(rest, convs, VAL_FRAC / (1 - TEST_FRAC), seed)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_heldout_theme(convs: dict, seed: int) -> dict:
    """Leave-one-theme-out: 7 folds, each holding out one theme as test."""
    folds = {}
    for theme in sorted({c["theme"] for c in convs.values()}):
        test = sorted(p for p, c in convs.items() if c["theme"] == theme)
        rest = sorted(p for p, c in convs.items() if c["theme"] != theme)
        train, val = stratified_split(rest, convs, VAL_FRAC, seed)
        folds[theme] = {"train": sorted(train), "val": sorted(val), "test": test}
    return folds


def stats(pids, convs) -> dict:
    rows = sum(convs[p]["n_rows"] for p in pids)
    met = sum(convs[p]["n_met"] for p in pids)
    dis = sum(convs[p]["n_disagree"] for p in pids)
    return {
        "conversations": len(pids),
        "rows": rows,
        "pct_met": round(100 * met / rows, 1),
        "not_met_rows": rows - met,
        "pct_disagree": round(100 * dis / rows, 1),
    }


def source_sha256() -> str | None:
    if not MANIFEST.exists():
        return None
    return json.loads(MANIFEST.read_text()).get("meta_eval.jsonl", {}).get("sha256")


def write(name: str, payload: dict, convs: dict, seed: int, counts: dict) -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLIT_DIR / f"{name}.json"
    out.write_text(
        json.dumps(
            {
                "name": name,
                "created": date.today().isoformat(),
                "seed": seed,
                "grouped_on": "prompt_id",
                "stratified_on": "cluster_set",
                "source_file": "data/raw/meta_eval.jsonl",
                "source_sha256": source_sha256(),
                "counts": counts,
                "splits": payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out


def verify_standard(s: dict, convs: dict) -> None:
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "prompt_id overlap"
    assert len(tr | va | te) == len(convs), "conversations missing from splits"
    total = sum(convs[p]["n_rows"] for p in convs)
    assert sum(stats(v, convs)["rows"] for v in s.values()) == total, "row count"


def verify_heldout(folds: dict, convs: dict, crit_by_pid: dict) -> None:
    for theme, s in folds.items():
        tr, te = set(s["train"]), set(s["test"])
        assert not (tr & te), f"{theme}: prompt_id overlap"
        assert not (set(s["val"]) & te), f"{theme}: val/test overlap"
        tr_c = set().union(*(crit_by_pid[p] for p in tr))
        te_c = set().union(*(crit_by_pid[p] for p in te))
        assert not (tr_c & te_c), f"{theme}: {len(tr_c & te_c)} criteria leak into test"


def criteria_by_pid() -> dict[str, set]:
    out = collections.defaultdict(set)
    with (RAW_DIR / "meta_eval.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]].add(r["rubric"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    convs = load_conversations()
    print(f"{len(convs):,} conversations, "
          f"{len({c['cluster_set'] for c in convs.values()})} strata, "
          f"{len({c['theme'] for c in convs.values()})} themes\n")

    std = build_standard(convs, args.seed)
    verify_standard(std, convs)
    counts = {k: stats(v, convs) for k, v in std.items()}
    print("=== SPLIT A: standard ===")
    print(f"{'split':<8}{'convos':>9}{'rows':>9}{'%met':>8}{'not-met':>9}{'%disagree':>11}")
    for k, c in counts.items():
        print(f"{k:<8}{c['conversations']:>9,}{c['rows']:>9,}"
              f"{c['pct_met']:>8}{c['not_met_rows']:>9,}{c['pct_disagree']:>11}")
    p = write("standard", std, convs, args.seed, counts)
    print(f"-> {p.relative_to(DATA_DIR.parent)}")

    print("\n  per-stratum conversations (train/val/test):")
    for s in sorted({c["cluster_set"] for c in convs.values()}):
        row = [sum(1 for x in std[k] if convs[x]["cluster_set"] == s)
               for k in ("train", "val", "test")]
        print(f"    {row[0]:>5}/{row[1]:>4}/{row[2]:>4}  {s.split('|')[0][:52]}")

    crit = criteria_by_pid()
    folds = build_heldout_theme(convs, args.seed)
    verify_heldout(folds, convs, crit)
    fcounts = {t: {k: stats(v, convs) for k, v in s.items()} for t, s in folds.items()}
    print("\n=== SPLIT B: heldout_theme (7 folds) ===")
    print(f"{'held-out theme':<22}{'train':>8}{'val':>7}{'test':>7}{'test rows':>11}{'criteria':>10}")
    for t, s in folds.items():
        n_crit = len(set().union(*(crit[p] for p in s["test"])))
        print(f"{t:<22}{len(s['train']):>8,}{len(s['val']):>7,}"
              f"{len(s['test']):>7,}{fcounts[t]['test']['rows']:>11,}{n_crit:>10}")
    p = write("heldout_theme", folds, convs, args.seed, fcounts)
    print(f"-> {p.relative_to(DATA_DIR.parent)}")

    print("\nall assertions passed")


if __name__ == "__main__":
    main()
