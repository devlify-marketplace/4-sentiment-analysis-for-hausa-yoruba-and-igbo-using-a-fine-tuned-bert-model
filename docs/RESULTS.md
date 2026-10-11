# Topic 4 — Evaluation Results

> Results are generated from the actual saved-model run. Check the run metadata below; subset runs are not comparable to full-split runs.

- **Model:** `Davlan/afro-xlmr-base` fine-tuned for 2 epoch(s).
- **Selection:** highest validation macro-F1 (epoch 2).
- **Training rows:** 3000 total; 1000 cap per language.
- **Official test rows:** 2400 total; subset cap per language: 800.
- **Seed:** 42; **device:** cpu; **max length:** 96.

## Held-out test metrics

| Scope | n | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| All languages | 2400 | 0.5996 | 0.5878 | 0.5839 | 0.5841 | 0.5958 |
| Hausa | 800 | 0.6350 | 0.6376 | 0.6360 | 0.6319 | 0.6313 |
| Yoruba | 800 | 0.5950 | 0.5698 | 0.5656 | 0.5635 | 0.5951 |
| Igbo | 800 | 0.5687 | 0.5541 | 0.5223 | 0.5000 | 0.5332 |

## Per-class test scores

| Language | Sentiment | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Hausa | negative | 0.5513 | 0.6491 | 0.5962 | 265 |
| Hausa | neutral | 0.6311 | 0.4815 | 0.5462 | 270 |
| Hausa | positive | 0.7305 | 0.7774 | 0.7532 | 265 |
| Yoruba | negative | 0.4000 | 0.3793 | 0.3894 | 174 |
| Yoruba | neutral | 0.5710 | 0.7028 | 0.6301 | 286 |
| Yoruba | positive | 0.7385 | 0.6147 | 0.6709 | 340 |
| Igbo | negative | 0.5246 | 0.1561 | 0.2406 | 205 |
| Igbo | neutral | 0.5873 | 0.7358 | 0.6532 | 352 |
| Igbo | positive | 0.5503 | 0.6749 | 0.6063 | 243 |

## Interpretation

Use macro-F1 and per-language/per-class results to compare coverage; accuracy alone can hide class imbalance. These scores measure only the released NaijaSenti Twitter benchmark under the exact configuration above. They are not expected performance on current social media, private feedback, other domains or every speaker community. Refer to `DATA_PROVENANCE.md` and the project report for limitations.
