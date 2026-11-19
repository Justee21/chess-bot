# Blind judgments and frozen deployment

The published bot uses fine-tune 2, frozen on 2026-11-15 in `config/final.json`. Its checkpoint/table SHA-256 values are verified at startup by the UCI adapter. The calibration and temperature artifacts for fine-tune 2 exist and select Elo input 1500, temperature 1.2, and relative cutoff 0.05; the transcript's later uncertainty about whether fine-tune 2 had been calibrated is resolved by those saved artifacts.

## Focused trials

The local `data/blind-log.jsonl` contains these 20 judgments. Justin was the sole judge. Each trial asked whether the hidden opponent played like him; it was not a guess of which model was playing.

| Model | Yes | Trials | Observed proportion | Marginal Wilson 95% interval |
| --- | --- | --- | --- | --- |
| Base | 0 | 7 | 0% | 0–35.4% |
| Fine-tune 1 | 1 | 6 | 16.7% | 3.0–56.4% |
| Fine-tune 2 | 3 | 7 | 42.9% | 15.8–75.0% |

All local rows have result `*`. They record color and verdict but do not preserve moves, length, or completion, so these are play trials rather than verified full games. The author's session notes describe the trials as mostly opening/middlegame. That phase breakdown cannot be checked from the saved log.

These intervals assume independent Bernoulli trials with a stable judgment probability, an approximation for repeated judgments from one person. They do not correct for choosing a focused subset or for comparisons across models. Interval overlap alone is not a formal test of a difference. The sample does not establish a ranking between the fine-tunes or prove base can never resemble the author.

## Earlier session context

The author supplied an earlier assistant transcript reporting another 33 trials. In those earlier trials, fine-tune 1 passed 4/12 and fine-tune 2 passed 3/10. Combining the transcript's earlier counts with the saved focused counts gives 5/18 and 6/17, respectively. Those earlier raw judgments are not present in the current local log and have not been independently reconstructed. The focused subset should not be presented as the only trials conducted.

## Selection

Fine-tune 2 is a design choice based on cleaned training data, approximately tied held-out prediction (63.9% versus 63.8%), and a slight subjective lead in the focused subset. It is not a statistically established winner over fine-tune 1. No additional trials or new training were required before releasing this version.

The held-out prediction comparison uses Elo input 1700, temperature 1, no position table, and the scramble-filtered test positions. Deployment uses Elo input 1500, temperature 1.2, relative cutoff 0.05, and the frozen personal table. These are different measurements of the same checkpoint; the deployment's top-1 score at its exact settings has not been reported.
