# Small, Robust LLM Judges for Health Chatbots

**NLP Final Project, Fall 2026**
Aneesh Sai Kolukuluri · Aly Mohamed Adel Mohamed Mahmoud · Siddharth Mehta · Michael O'Bannon

## Overview

HealthBench (Arora et al., 2025) grades health-chatbot responses against physician-written rubric criteria using a large API model as the judge. That judge agrees with physicians about as often as physicians agree with each other, but it is expensive, and it requires sending health conversations to an outside service.

We ask whether a **small (≤3B parameter) judge, trained on HealthBench physician labels**, can match the large API judge's agreement with physicians. We also ask whether training on soft labels from physician disagreement makes it better calibrated. A judge this small could run locally, keep patient data private, and cost very little to run.

## Research Questions

Priorities follow the instructor's feedback.

| RQ | Question | Priority |
|----|----------|----------|
| **RQ1** | Can a sub-3B judge trained on physician labels approach the physician agreement of a large API judge? | **Core** |
| **RQ3** | Does training on soft labels from multiple physicians (they disagree on ~22.5% of cases) give a better-calibrated judge? | **Core** |
| **RQ4** | When trained the same way, how do a long-context encoder and a small decoder LLM compare on agreement, robustness and cost? | **Core** |
| RQ2 | Do counterfactual augmentation and an invariance loss make the judge robust to unseen rewording, keeping its verdict under meaning-preserving edits and flipping it under meaning-changing medical edits? | Stretch, only if time allows (see note) |
| RQ5 | Would ranking chatbots with our small judge instead of the API judge lead to the same conclusions about model quality and safety? | Stretch, unlikely |

> **Note on RQ2:** A naive edit such as swapping a dose does not always change whether a rubric criterion is met. Every meaning-changing edit has to be checked against the specific criterion it targets. Before investing in RQ2, talk to Karan, Siham and Mohammed's group, who are working on counterfactuals.

## Data

- **HealthBench** ([openai/simple-evals](https://github.com/openai/simple-evals)): 5,000 multi-turn health conversations with physician-written rubric criteria.
- **Meta-evaluation labels:** 2–5 physician "met / not met" judgments per (response, criterion). These give us training and test labels, soft labels (RQ3) and the physician–physician agreement ceiling.
- **Consensus subset:** high-agreement criteria, used as a clean test set.
- **Counterfactual edit set** (RQ2 only, built by us): rule-based edits and constrained LLM rewrites, filtered with bidirectional NLI and checked by hand.

Raw data is **not committed** to this repo. See `data/README.md` for download instructions.

## Method

1. **Baselines:** the large API judge, zero-shot open models (1–3B general, 8B general, medical fine-tuned), and a surface-features-only classifier.
2. **Encoder judge** (Siddharth): a long-context encoder fine-tuned as a classifier on (conversation, response, criterion) → met / not met.
3. **Decoder judge** (Michael): a small decoder LLM fine-tuned with LoRA.
4. **Soft labels** (RQ3): train on the fraction of physicians who said "met" instead of the majority vote.
5. **Consistency loss** (RQ2, stretch): `L = CE(p(x), y) + λ · KL(p(x) || p(x'))`, where `x'` is an edited version of `x`.

## Evaluation

- **Agreement:** macro-F1, Cohen's κ, and the judge's percentile within physician–physician agreement, reported per axis and per theme.
- **Calibration:** ECE, Brier score, reliability diagrams.
- **Robustness** (RQ2): flip rate on meaning-preserving edits and detection rate on meaning-changing edits, per edit family.
- **Cost:** latency and dollars per 1k judgments.
- **Statistics:** paired bootstrap confidence intervals and McNemar's test.
