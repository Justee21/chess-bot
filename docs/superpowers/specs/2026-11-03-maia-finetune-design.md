# Fine-tune Maia 3 on Justin's games: design

Date: 2026-11-03. Builds on `2026-11-02-chess-bot-design.md` (harness, evaluation, data pipeline all exist and are reused).

## Intent and why

The harness (opening table + plain Maia 3) matches Justin's opening moves well (70% top-1 in the opening) but in the middlegame and endgame it is plain Maia: 57.8% top-1 on held-out games, 70% of Justin's moves. Playing it, Justin sees moves he would never play. Example: King's Indian structure, Black to move; Justin plays ...e5 "without thinking" (he has played it there), Maia gives e5 38.5% and the bot sampled ...Re8 (7.2%). The bot plays low-probability moves about as often as Justin does (14.7% vs 13.4% of moves under 8% Maia probability), but they are random, not his habits. A fine-tune should teach Maia his habits so they carry across similar positions, which an exact-position table cannot do.

## Goals

- A Maia 3 checkpoint fine-tuned on Justin's moves from the training split, that plays more like him than base Maia, in the middlegame and endgame especially.
- Base and fine-tuned models both stay available and are directly comparable: same held-out test, same metrics, and in play.
- Same strength as before: Justin's lifetime-average level (average centipawn loss near his 68.8 train / 71.4 test).

## Non-goals

- Hosting, ONNX export, the LLM playstyle judge, the Lichess bot (all still deferred).
- Training from scratch, or changing the table, the no-resign rule or the no-draw rule.
- Matching Justin's current strength instead of his lifetime average (his decision).
- Clock or time-pressure modelling (the bot has no clock).

## Anti-goals (a failure even if numbers look fine)

- Memorizing training games: better on training positions, no better or worse on held-out ones.
- A bot that is noticeably stronger or weaker than Justin, or that plays illegal or silly moves.
- Overwriting or losing the base model.
- Any test-split game used in training, validation, early stopping or tuning.

## Decided by Justin

- Maia 3 is the base; fine-tune it (the 23M model is the one in use).
- Keep base and fine-tuned models side by side for comparison.
- Data is Justin0921's chess.com games across all periods and time controls, split by date as before (oldest 85% train, newest 15% test).
- "Plays my playstyle" is the aim; eras differ and that is known and accepted.
- No look at era data is needed first.

## My calls (Justin can overrule)

- **Targets:** only Justin's own moves are training targets, as in the table. Opponent moves are never targets.
- **Rating inputs:** each training position uses Justin's rating in that game as the self Elo and the opponent's rating as the opponent Elo, so the era differences are explained by the rating input instead of blurred together. At play time the calibrated lifetime-average Elo (1700, to be recalibrated after the fine-tune) is used for both, as now.
- **Time class:** if the model's inputs allow it, tag bullet/blitz/rapid; otherwise down-weight bullet moves. Left to the builder after reading the model code.
- **Gentle training:** low learning rate, few epochs, early stopping on a validation slice taken from the NEWEST part of the training split (never the test split). Consider freezing early layers, or a loss term keeping the output close to base Maia so it stays sane. The builder chooses and records what worked.
- **Where it runs:** Google Colab Pro GPU (already paid for), with the training script and a small JSON of Justin's training positions. The 23M model also likely trains on the Mac as a fallback. Output is a `.pt` file saved separately (about 92 MB), never over the base weights.
- **After the fine-tune:** re-run Elo calibration and temperature tuning on the fine-tuned model, since it will be sharper and the old 1700 / 1.0 may no longer match Justin's error profile.
- **Training code:** the public maia3 repo has no training code, so a training loop is written here. First step is a spike: confirm the model runs forward with gradients on a small batch and that the loss drops on a few training positions.

## Success measures (all on the untouched test split)

Compared with base Maia and the current harness, same evaluation tool:

1. Middlegame and endgame top-1 and mean log-probability. Baseline 57.8% top-1 in both. Target: a clear improvement (my bar: at least +2 points top-1 and better mean log-probability), with the opening not worse.
2. Error profile (Stockfish depth 12, self-play): average centipawn loss within about 10% of Justin's (68.8 train sample), p100/p200/p300 in the same range, after re-tuning temperature.
3. Overfitting check: training-set vs validation accuracy gap reported; early stop where validation stops improving.
4. Justin plays it against the base version and says which feels more like him (final judge). A side-by-side view of base vs fine-tuned top moves for a position helps (the KID e5 / Re8 position is the first test case).
5. Later: the LLM playstyle judge.

## Constraints

- AGPL-3.0 applies (repo already carries it); derived weights stay with the same licence.
- Zero extra cost: free Colab Pro quota or the Mac.
- Existing tests keep passing; new code is test-first.
- The no-draw and no-resign rules, and the table-first behavior, are unchanged.

## What the builder decides

Training loop details, layers frozen or not, learning rate and epochs, batch size, loss regularizer, the time-class handling, the checkpoint format and how `--model` / a `--checkpoint` flag selects base or fine-tuned in `evaluate`, `play`, `calibrate` and `tune-temperature`, and how the side-by-side position view is shown.

## Risks

- About 8,500 train games is small for 23M parameters, so overfitting is the main risk.
- Over half the games are bullet, which may teach fast, low-quality habits.
- The model may not expose a clean loss or may need small code changes to train; the spike finds out.
