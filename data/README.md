# Data

Raw files are **not committed**. Download them with:

```bash
python data/download.py
```

That writes four files into `data/raw/` and records sizes and SHA-256 digests in
`data/manifest.json`. The manifest **is** committed, so we can confirm we all
trained on the same bytes — the upstream URLs carry dates and have been reissued
before. Use `--only meta_eval.jsonl` to fetch a subset, `--force` to re-download.

| Local file | Size | Contents |
|---|---|---|
| `raw/meta_eval.jsonl` | 130 MB | Physician met/not-met labels. **The only file with human labels.** |
| `raw/eval.jsonl` | 57 MB | Main HealthBench: conversations + full rubrics |
| `raw/consensus.jsonl` | 35 MB | Consensus subset |
| `raw/hard.jsonl` | 12 MB | Hard subset |

All four come from `https://openaipublic.blob.core.windows.net/simple-evals/healthbench/`.
The exact remote filenames live in `download.py`.

## meta_eval.jsonl — the training set for RQ1 and RQ3

One row = one (conversation, response, criterion) with a few physician votes.

| Field | Contents |
|---|---|
| `prompt` | list of `{role, content}` — the conversation |
| `completion` | the assistant response under judgment |
| `rubric` | the grading criterion, as a long instruction string |
| `binary_labels` | list of bools, one per physician |
| `anonymized_physician_ids` | parallel list of physician IDs |
| `category` | cluster name, e.g. `cluster:emergency_referrals_emergent_emergency_behavior` |
| `prompt_id` | conversation ID — **group splits on this** |
| `completion_id` | conversation+response ID |
| `canary` | contamination-detection GUID; strip before training |

### Measured shape

| | |
|---|---|
| Rows | 29,511 |
| Physician observations | 60,896 |
| Unique conversations (`prompt_id`) | 3,671 |
| Unique (conversation, response) | 14,591 |
| **Unique rubric criteria** | **30** |
| Unique categories | 34 |
| Unique physicians | 186 |
| Conversation turns | median 1, max 19 |

Physicians per row: **2 (94.1%)**, 3 (5.6%), 4 (0.2%), 5 (0.1%).

Class balance: **met 87.5% / not-met 12.5%**.

Physician disagreement: **22.5%** of rows.

Soft label = `mean(binary_labels)`: `1.0` 65.8%, `0.5` 20.4%, `0.0` 11.6% —
**97.8% of rows take one of three values.**

## Four things these numbers change

**1. The criteria are a closed set of 30.** This is a meta-evaluation set, not
per-conversation rubrics: the same 30 cluster-level grading instructions are
reused across all 29,511 rows. A model can learn each criterion's base rate and
score well without reading the response. Report a **held-out-criterion split**
alongside the main one, or we cannot claim the judge generalizes to new rubrics.

**2. Split on `prompt_id`, never on rows.** Each conversation appears in ~8 rows
(≈4 responses × ≈2 criteria). A random row split leaks the same conversation
into train and test.

**3. The classes are heavily imbalanced.** Always-predict-met scores 87.5%
accuracy, so accuracy is meaningless here. Macro-F1 and Cohen's κ are the
headline metrics, and the majority-class baseline must be reported next to them.

**4. Soft labels are three-valued.** With 94% of rows judged by exactly two
physicians, RQ3's "soft labels" are effectively `{0, 0.5, 1}` — closer to
three-way label smoothing than a fine-grained uncertainty distribution. Still
worth testing, but the claim has to be stated at that resolution.

## Physician agreement ceiling

Only the 5.9% of rows with ≥3 physicians support leave-one-physician-out
agreement. On 2-physician rows the only available statistic is raw pairwise
agreement (77.5%). The ceiling for RQ1 has to be computed and described
carefully; see `eval/` once the harness lands.
