"""
Turns dataset rows into encoder input IDs: [CLS] criterion [SEP] completion [SEP] conversation [SEP], 
truncated to MAX_LEN (see notebooks/encoder_lengths.ipynb).
"""

from __future__ import annotations

import pandas as pd

MAX_LEN = 2048
VARIANTS = ("full", "last_turn", "no_criterion")



def last_user_turn(conversation: str) -> str:

    """Only the final user message. Approximate: splits on the rendered role prefix."""

    head, sep, tail = conversation.rpartition("\nuser: ")

    if sep:
        return "user: " + tail

    return conversation


def pack(crit: list[int], comp: list[int], conv: list[int], cls: int, sep: int, max_len: int = MAX_LEN) -> list[int]:

    """[CLS] crit [SEP] comp [SEP] conv [SEP], truncated to max_len."""
    
    head = [cls] + (crit + [sep] if crit else [])
    
    n_special = len(head) - len(crit) + 2 # the [SEP] after comp and after conv

    room = max_len - n_special - len(crit) - len(comp)

    if room >= len(conv):
        pass # fits whole
    elif room > 0:
        conv = conv[-room:]# keep the latest turns
    else:
        conv = [] # no room: drop conversation, cut completion's end
        comp = comp[: max_len - n_special - len(crit)]

    return head + comp + [sep] + conv + [sep]


def _tokenize_unique(col: pd.Series, tok) -> list[list[int]]:

    unique = col.unique().tolist()

    ids = tok(unique, add_special_tokens=False)["input_ids"]

    lookup = dict(zip(unique, ids))

    return [lookup[s] for s in col]


def encode_frame(df: pd.DataFrame, tok, variant: str = "full", max_len: int = MAX_LEN) -> list[list[int]]:

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    conv_text = df.conversation

    if variant == "last_turn":

        conv_text = conv_text.map(last_user_turn)

    comp = _tokenize_unique(df.completion, tok)
    conv = _tokenize_unique(conv_text, tok)

    crit = ([[] for _ in range(len(df))] if variant == "no_criterion" else _tokenize_unique(df.criterion, tok))

    return [pack(c, m, v, tok.cls_token_id, tok.sep_token_id, max_len) for c, m, v in zip(crit, comp, conv)]

if __name__ == "__main__":

    from transformers import AutoTokenizer
    from data.load import load_split

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train, _, _ = load_split("standard")

    for variant in VARIANTS:

        ids = encode_frame(train, tok, variant)

        lens = pd.Series(map(len, ids))

        print(f"{variant:<13} max {lens.max():>5} at cap {100 * (lens == MAX_LEN).mean():5.2f}%")

        assert lens.max() <= MAX_LEN

    print("\n" + tok.decode(encode_frame(train.head(1), tok)[0])[:600])
