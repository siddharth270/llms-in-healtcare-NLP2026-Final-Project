#!/usr/bin/env python3
"""Agreement metrics for the HealthBench judge.

Mirrors healthbench_meta_eval.py: physician labels are never collapsed into a
consensus. A prediction is compared against each physician separately, and the
headline number is F1 averaged over the met and not-met classes
(`pairwise_model_f1_balanced` upstream).

The same routine gives the physician ceiling -- treat one physician as the
prediction and the others as reference -- so the judge's score and the human bar
it is measured against come from one implementation.
"""

from __future__ import annotations

import numpy as np

MET = "met"
NOT_MET = "not_met"


def _f1(pred: np.ndarray, ref: np.ndarray, positive: bool) -> float:
    """F1 over paired boolean arrays, with `positive` as the positive class."""
    p = pred == positive
    r = ref == positive
    tp = int(np.sum(p & r))
    fp = int(np.sum(p & ~r))
    fn = int(np.sum(~p & r))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _expand(probs, binary_labels, threshold: float):
    """One (prediction, physician_label) pair per physician per row."""
    pred, ref = [], []
    for p, labels in zip(probs, binary_labels):
        decision = bool(p >= threshold)
        for label in labels:
            pred.append(decision)
            ref.append(bool(label))
    return np.array(pred), np.array(ref)


def _scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    f1_pos = _f1(pred, ref, True)
    f1_neg = _f1(pred, ref, False)
    return {
        "f1_balanced": (f1_pos + f1_neg) / 2,
        f"f1_{MET}": f1_pos,
        f"f1_{NOT_MET}": f1_neg,
        # reported only to show how misleading it is at 87.5% base rate
        "accuracy": float(np.mean(pred == ref)) if len(pred) else 0.0,
        "n_pairs": int(len(pred)),
    }


def pairwise_f1_balanced(probs, binary_labels, threshold: float = 0.5) -> dict:
    """Score predictions against every physician individually."""
    return _scores(*_expand(probs, binary_labels, threshold))


def physician_ceiling(binary_labels) -> dict:
    """Physician-vs-physician agreement: the bar the judge is measured against.

    Every ordered pair of physicians on a row contributes one comparison, so
    rows judged by only two physicians still count -- which is 94% of them.
    """
    pred, ref = [], []
    for labels in binary_labels:
        if len(labels) < 2:
            continue
        for i, self_label in enumerate(labels):
            for j, other in enumerate(labels):
                if i != j:
                    pred.append(bool(self_label))
                    ref.append(bool(other))
    return _scores(np.array(pred), np.array(ref))


def evaluate(df, probs, threshold: float = 0.5) -> dict:
    """Headline score plus the breakdowns the evaluation plan calls for.

    `df` needs columns binary_labels, theme, axis and unanimous; `probs` is one
    probability per row, aligned to df's order.
    """
    probs = np.asarray(probs, dtype=float)
    assert len(probs) == len(df), f"{len(probs)} probs vs {len(df)} rows"

    out = {
        "overall": pairwise_f1_balanced(probs, df.binary_labels, threshold),
        "ceiling": physician_ceiling(df.binary_labels),
        "by_theme": {},
        "by_axis": {},
    }

    # Split C: the same test set with physician-ambiguous rows removed, so the
    # gap against `overall` isolates how much error was label noise
    mask = df.unanimous.to_numpy()
    out["unanimous_only"] = pairwise_f1_balanced(
        probs[mask], df.binary_labels[mask], threshold
    )

    for field in ("theme", "axis"):
        for value, group in df.groupby(field, sort=True):
            idx = df.index.get_indexer(group.index)
            out[f"by_{field}"][value] = pairwise_f1_balanced(
                probs[idx], group.binary_labels, threshold
            )
    return out


def format_report(result: dict) -> str:
    """Human-readable table. Row counts sit beside every score on purpose."""
    lines = []

    def row(name, s):
        lines.append(
            f"  {name:<24}{s['f1_balanced']:>8.3f}{s[f'f1_{MET}']:>9.3f}"
            f"{s[f'f1_{NOT_MET}']:>10.3f}{s['accuracy']:>10.3f}{s['n_pairs']:>9,}"
        )

    lines.append(f"  {'':<24}{'f1_bal':>8}{'f1_met':>9}{'f1_not':>10}{'acc':>10}{'pairs':>9}")
    row("OVERALL", result["overall"])
    row("physician ceiling", result["ceiling"])
    row("unanimous rows only", result["unanimous_only"])
    for field in ("theme", "axis"):
        lines.append(f"\n  by {field}:")
        for k, v in result[f"by_{field}"].items():
            row(k, v)
    return "\n".join(lines)
