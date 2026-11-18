"""
Fine-tune a long context encoder as a met/not-met judge.

python -m encoder.train --target hard --variant full --seed 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from data.load import load_split
from encoder.inputs import MAX_LEN, VARIANTS, encode_frame
from eval.metrics import pairwise_f1_balanced


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

def parse_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--target", choices=("hard", "soft"), default="hard")
    ap.add_argument("--variant", choices=VARIANTS, default="full")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--neg-weight", type=float, default=1.0, help="loss weight on the not-met side; >1 counters the 87.5%% met imbalance")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="sample N train and N val rows, for a quick smoke test")
    ap.add_argument("--out", type=Path, default=None)

    return ap.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 on Ampere or newer (A100/H100/L40), fp16 on V100/P100, None off CUDA.

    Checks compute capability rather than torch.cuda.is_bf16_supported(), which also
    returns True for slow emulated bf16 on a V100.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(ids: list[list[int]], order: list[int], size: int, pad_id: int):
    """Yield (row indices, input_ids, attention_mask), padded to the longest row in the batch."""

    for start in range(0, len(order), size):

        idx = order[start:start + size]

        width = max(len(ids[i]) for i in idx)

        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)

        for row, i in enumerate(idx):

            input_ids[row, :len(ids[i])] = torch.tensor(ids[i])
            mask[row, :len(ids[i])] = 1

        yield idx, input_ids, mask



@torch.no_grad()
def predict_probs(model, ids, batch_size, pad_id, device, amp) -> np.ndarray:
    """P(met) per row, in the same order as ids."""

    model.eval()
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    probs = np.empty(len(ids))

    for idx, x, m in batches(ids, order, batch_size, pad_id):

        with amp():

            logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.squeeze(-1)
        
        probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()

    model.train()

    return probs


def best_threshold_f1(probs, binary_labels) -> tuple[float, float, float]:
    """(best threshold, F1 at it, F1 at 0.5) on val, scored against every physician."""
    scores = {t: pairwise_f1_balanced(probs, binary_labels, t)["f1_balanced"] for t in THRESHOLDS}
    best = max(scores, key=scores.get)

    return best, scores[best], scores[0.5]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()


    dtype = amp_dtype(device)
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if dtype else contextlib.nullcontext
    # fp16 needs loss scaling or small gradients round to zero; bf16 and fp32 do not
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    out = args.out or ROOT / "checkpoints" / "encoder" / f"{args.variant}_{args.target}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    train, val, _ = load_split("standard")
   
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=args.seed).reset_index(drop=True)
        val = val.sample(min(args.limit, len(val)), random_state=args.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    train_ids = encode_frame(train, tok, args.variant, args.max_len)
    val_ids = encode_frame(val, tok, args.variant, args.max_len)

    # the only line that differs between the hard and soft runs
    col = "hard_label" if args.target == "hard" else "soft_label"
    y = torch.tensor(train[col].astype(float).to_numpy(), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_batches = math.ceil(len(train_ids) / args.batch_size)
    total_steps = math.ceil(n_batches / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)

    # Training Loop

    best, history, step, t0 = {"val_f1": -1.0}, [], 0, time.time()

    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(train_ids)).tolist()
        running = 0.0
        for k, (idx, x, m) in enumerate(batches(train_ids, order, args.batch_size, tok.pad_token_id), 1):
            with amp():
                logits = model(input_ids=x.to(device), attention_mask=m.to(device)).logits.squeeze(-1)
            target = y[idx].to(device)
            # weight 1 on met, neg_weight on not-met; a 0.5 soft label lands halfway
            weight = 1 + (args.neg_weight - 1) * (1 - target)
            loss = binary_cross_entropy_with_logits(logits.float(), target, weight=weight)
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()

            if k % args.grad_accum == 0 or k == n_batches:
                scaler.unscale_(opt)                 # so clipping sees the real gradient size
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch}  step {step}/{total_steps}  loss {running / k:.4f}", flush=True)

        probs = predict_probs(model, val_ids, args.batch_size * 2, tok.pad_token_id, device, amp)
        t, f1_t, f1_05 = best_threshold_f1(probs, val.binary_labels)
        history.append({"epoch": epoch, "train_loss": running / n_batches,
                        "val_f1": f1_t, "val_threshold": t, "val_f1_at_0.5": f1_05})
        print(history[-1], flush=True)

        if f1_t > best["val_f1"]:
            best = history[-1]
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")

    run = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": str(dtype).removeprefix("torch.") if dtype else "fp32",
        "best": best,
        "history": history,
        "train_seconds": round(time.time() - t0, 1),     # includes the per-epoch val passes
        "n_params": sum(p.numel() for p in model.parameters()),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "train_rows": len(train_ids),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(f"\nbest epoch {best['epoch']}  val F1 {best['val_f1']:.3f} @ {best['val_threshold']}  -> {out}")


if __name__ == "__main__":
    main()
