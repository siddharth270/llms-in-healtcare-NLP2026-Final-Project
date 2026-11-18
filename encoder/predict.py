"""Score a trained encoder on test and write the shared prediction file.

Reads the run's settings and val threshold from run.json, so predict always matches train.

    python -m encoder.predict checkpoints/encoder/full_hard_s1
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.load import load_split
from encoder.inputs import encode_frame
from encoder.train import ROOT, amp_dtype, pick_device, predict_probs
from eval.metrics import evaluate, format_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    run = json.loads((args.run_dir / "run.json").read_text())
    cfg, threshold = run["args"], run["best"]["val_threshold"]

    device = pick_device()
    dtype = amp_dtype(device)
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else contextlib.nullcontext
    tok = AutoTokenizer.from_pretrained(args.run_dir / "best")
    model = AutoModelForSequenceClassification.from_pretrained(args.run_dir / "best").to(device)

    _, _, test = load_split("standard")
    ids = encode_frame(test, tok, cfg["variant"], cfg["max_len"])
    probs = predict_probs(model, ids, args.batch_size, tok.pad_token_id, device, amp)

    # same schema as open_judges/open_judge.py write_standard_predictions
    name = args.run_dir.name
    out = ROOT / "results" / f"preds_encoder_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "completion_id": test.completion_id.astype(str),
        "criterion": test.criterion.astype(str),
        "p_met": probs,
    }).to_parquet(out, index=False)

    metrics = {
        "run": name,
        "val_threshold": threshold,
        "at_0.5": evaluate(test, probs, 0.5)["overall"],
        "at_val_threshold": evaluate(test, probs, threshold),
        # calibration against the physician vote share, for RQ3
        "brier_vs_soft": float(np.mean((probs - test.soft_label.to_numpy()) ** 2)),
    }
    (ROOT / "results" / f"metrics_encoder_{name}.json").write_text(json.dumps(metrics, indent=2))

    print(f"test @ 0.5:  f1_bal {metrics['at_0.5']['f1_balanced']:.3f}")
    print(f"test @ {threshold} (val):\n{format_report(metrics['at_val_threshold'])}")
    print(f"Brier vs soft label: {metrics['brier_vs_soft']:.4f}   -> {out}")


if __name__ == "__main__":
    main()
