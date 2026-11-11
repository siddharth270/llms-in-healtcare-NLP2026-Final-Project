# Split design

Spec for `data/build_splits.py`. Read this before coding it. Every number below
was measured from `data/raw/meta_eval.jsonl`; see `data/README.md` for the full
dataset characterisation.

## Scope

Three split definitions, all defined over **conversations** (`prompt_id`), never
over rows:

| Name | Purpose | Answers |
|---|---|---|
| **A — standard** | main experiments (RQ1, RQ3, RQ4) | how good is the judge? |
| **B — heldout_theme** | generalisation | did it learn, or memorise 30 criteria? |
| **C — clean test** | label-noise control | how much error is physician disagreement? |

## Input

`data/processed/meta_eval.parquet` (from `build_dataset.py`), or the raw JSONL.
Only three fields are needed to build splits: `prompt_id`, `category`, `theme`.

Build a **conversation table** first — one row per `prompt_id`:

| Column | Value |
|---|---|
| `prompt_id` | the conversation ID |
| `cluster_set` | its set of `category` values, sorted and joined into one string |
| `theme` | its theme (every conversation has exactly one) |

Expect **3,671 conversations**, **17 distinct `cluster_set` values**, **7 themes**.

## Output

`data/splits/standard.json` and `data/splits/heldout_theme.json`, committed to
git. Schema:

```json
{
  "name": "standard",
  "created": "2026-11-11",
  "seed": 20261111,
  "grouped_on": "prompt_id",
  "stratified_on": "cluster_set",
  "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
  "source_file": "data/raw/meta_eval.jsonl",
  "source_sha256": "<copy from data/manifest.json>",
  "counts": {"train": {"conversations": 2569, "rows": 20657}, "...": {}},
  "splits": {"train": ["<prompt_id>", "..."], "val": [], "test": []}
}
```

The `source_sha256` is the point of the manifest: it proves every teammate split
the same bytes.

## Split A — standard

**70 / 15 / 15 over conversations, stratified on `cluster_set`.**

Expected: **2,569 train / 551 val / 551 test** conversations
(~20,700 / ~4,400 / ~4,400 rows).

Algorithm — two-stage stratified split:

```
1. test        = 15% of conversations,  stratify on cluster_set
2. train, val  = split the remaining 85%, val = 0.1765 of it,
                 stratify on cluster_set again
```

`sklearn.model_selection.train_test_split` with `stratify=` and a fixed
`random_state` handles both stages. No grouping machinery is needed — each unit
is already one conversation, so grouping is satisfied by construction.

### Why

- **Grouped on conversation.** Each conversation yields ~8 rows (4 responses ×
  ~2 criteria), and responses to the same prompt are often near-identical. A
  row-level split puts paraphrases of training items into test.
- **Stratified on `cluster_set`, not on individual clusters.** Each conversation
  has exactly one `cluster_set` and there are only 17, so this is a plain
  single-label stratification. The smallest stratum has 134 conversations, so
  even a 15% slice yields ~20.
- **Not stratified on agreement.** Disagreement is a row property but is largely
  determined by cluster (9%–45% by cluster), so balancing clusters balances
  agreement as a side effect. Verify it afterwards instead.
- **A val set exists** so that early stopping and hyperparameter choice never
  touch test. Val is the only tuning surface.

## Split B — heldout_theme

**Leave-one-theme-out: 7 folds.** Each fold trains on 6 themes, tests on the 7th.
Carve a val set as 15% of the 6 training themes, stratified on `cluster_set`.

| Theme | Conversations | Rows | Criteria |
|---|---|---|---|
| hedging | 711 | 8,526 | 6 |
| communication | 746 | 5,922 | 4 |
| global_health | 634 | 2,515 | 3 |
| emergency_referrals | 453 | 3,612 | 6 |
| context_seeking | 408 | 3,230 | 4 |
| health_data_tasks | 395 | 3,114 | 4 |
| complex_responses | 324 | 2,592 | 3 |

### Why

With only 30 criteria, a model can learn each criterion's base rate and score
well on Split A **without reading the response**. Split A cannot detect this,
because its test criteria are criteria it trained on. Split B can.

Themes are the right unit because they partition the data cleanly on both axes:
conversations sum to exactly 3,671 and criteria to exactly 30, with **zero
criteria shared across themes** (verified). Holding out individual criteria
instead would split conversations apart — a conversation with criteria `{A, B}`
would land partly in train and partly in test.

All 7 folds, not one: training takes minutes, and 7 folds give a mean and spread
rather than one number that may be an artefact of which theme was chosen.

**State precisely in the paper** that this measures transfer to an unseen task
type, not robustness to rephrasing. The 30 criteria map ~1:1 onto the 34
clusters, so held-out-criterion is really held-out-cluster.

## Split C — clean test

**Not a separate split.** A filter on Split A's test set: rows where all
physicians agreed (`soft_label` is 0.0 or 1.0). Expect ~77% of rows to survive,
so ~3,400 of ~4,400 test rows.

Implement as a boolean mask the harness can apply, not as its own ID list.

### Why

A separate clean sample would contain different conversations, confounding the
comparison. As a strict subset, scoring the same test set twice isolates exactly
one variable:

```
macro-F1, all test rows        → includes irreducibly ambiguous items
macro-F1, unanimous rows only  → label noise removed
gap                            → how much error was physician disagreement
```

## Verification

Assertions that must pass before the splits are committed:

- Split A: zero `prompt_id` overlap between train / val / test
- Split A: conversations sum to 3,671; rows sum to 29,511
- Split B: every fold has zero `prompt_id` **and** zero `criterion` overlap
  between train and test
- Split C: a strict subset of Split A's test set

Tables to print (eyeball these, do not just assert):

- Per-stratum conversation counts across train / val / test — confirms
  stratification held
- Class balance per split — expect ~87.5% met everywhere
- Disagreement rate per split — expect ~22.5% everywhere
- **Not-met row count per split** — expect ~550 in test. This is the project's
  real statistical power and it will look uncomfortably small.

If class balance or disagreement rate drifts more than ~2 points between train
and test, stratification failed. Investigate before proceeding.

## Rules

1. **Fixed seed**, recorded in the JSON.
2. **Commit the ID lists.** Never regenerate after anyone starts training —
   re-rolling silently invalidates every comparison made before the change.
3. **One split definition for the whole project.** Soft vs hard labels (RQ3),
   encoder vs decoder (RQ4) and all baselines use the identical splits, or those
   comparisons mean nothing.

## Gotchas

- `cluster_set` must be **order-independent** — sort the categories before
  joining, or the same set produces different strata strings.
- Return train / val / test as **separate frames**, not one frame with a `split`
  column that a caller can forget to filter on.
- Splits are defined on conversations; expand to rows only after splitting.
