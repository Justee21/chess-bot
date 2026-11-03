# Maia fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Maia 3 (23M) on Justin's own moves, keep base and fine-tuned models side by side, and measure the fine-tune against base on the untouched test split.

**Architecture:** Training examples are built from Justin's training-split games (compact int8 board frames, his rating and the opponent's, the target move index, legal-move indices, a weight). A small training loop fine-tunes the maia3 `MAIA3Model` with a masked cross-entropy loss, early-stops on a validation slice cut from the newest part of the training split, and saves a separate `.pt` checkpoint. `MaiaPredictor` gains a `checkpoint` option so every existing command (`evaluate`, `play`, `calibrate`, `tune-temperature`, `book-depth`) can run base or fine-tuned. Training runs on Colab Pro; everything else runs locally.

**Tech Stack:** Python 3.12 (.venv), torch, maia3 0.1.0, python-chess, numpy, pytest. Existing `chessbot` package.

**Spec:** `docs/superpowers/specs/2026-11-03-maia-finetune-design.md` (builds on `docs/superpowers/specs/2026-11-02-chess-bot-design.md`).

## Global Constraints

- Training targets are Justin's own moves only (`board.turn == game.my_color`); opponent moves are never targets.
- Split by date as before: test = newest 15% of all games (never touched by training, validation, early stopping or tuning); validation = newest 10% of the remaining training games.
- Self Elo = `game.my_elo`, opponent Elo = `game.opp_elo` for each training position. At play time the calibrated Elo is used for both (existing behavior).
- The 23M model has `include_time_info=False`, so time control cannot be an input: bullet moves are down-weighted (default weight 0.5), other classes 1.0.
- Model input is `(B, 64, 96)` float tokens (8 history frames x 12 piece planes), `self_elo` and `oppo_elo` as `torch.long` of shape `(B,)`; `model(...)` returns a tuple whose first element is logits `(B, V)` over `get_all_possible_moves()`; board and moves are mirrored when Black is to move (maia3's `tokenize_board`, `mirror_move`).
- Base weights are never overwritten: the fine-tuned checkpoint is a new file `{"model_state_dict": ...}` (loadable by maia3's `load_model`), and saving refuses an existing path.
- No-draw and no-resign rules, table-first behavior and AGPL-3.0 licensing unchanged.
- Tests are written first and watched failing. Fast tests use stub models; tests that need real Maia are marked `@pytest.mark.slow` and use the 5M model. Run the whole suite with `.venv/bin/pytest -q -m "slow or not slow"`.
- Code reads like the surrounding code: type hints, short docstrings, `from __future__ import annotations`.

## Review Focus

- **Leakage:** no test-split game, or validation game, appears in training; validation games are all newer than every training game (the test pins this in Task 1; Task 5 pins it through the CLI).
- **Only Justin's moves, correct side:** a game where Justin is Black yields targets for Black's moves only, with the target index and legal indices mirrored (Task 2 test).
- **Promotions and rare moves:** a promotion target maps to a valid vocabulary index and is among the legal indices; a position with a single legal move still trains (Task 2 test).
- **Same input as inference:** the training encoding of a position equals what `MaiaPredictor` feeds the model, including short histories at the start of a game (Task 2 test).
- **Early stopping returns the best epoch, not the last** (Task 3 test), and a checkpoint reloads into `MaiaPredictor` with identical probabilities (Task 4 test).

---

### Task 1: Three-way split and base-model plumbing

**Files:**
- Modify: `src/chessbot/split.py`, `src/chessbot/maia.py`
- Test: `tests/test_split.py`, `tests/test_maia.py` (slow)

**Interfaces:**
- Produces: `three_way_split(games: list[Game], test_fraction: float = 0.15, val_fraction: float = 0.10) -> tuple[list[Game], list[Game], list[Game]]` returning `(train, val, test)`. `test` equals `split_by_date(games, test_fraction)[1]`; `val` is the newest `max(1, round(len(rest) * val_fraction))` of the rest; `train` is what remains. Raises `ValueError` if there are fewer than 3 games.
- Produces: `history_frames(board: chess.Board, history: int = 8) -> deque` in `maia.py` (module function, the body of the current `MaiaPredictor._history`, taking `history` instead of `self.cfg.history`); `MaiaPredictor._history` calls it.
- Produces: `MaiaPredictor(model: str = "maia3-23m", device: str = "cpu", checkpoint: Path | None = None)`: when `checkpoint` is given, it replaces `cfg.checkpoint_path` before `load_model`; a missing path raises `FileNotFoundError`.

- [ ] **Step 1: Commit the pending work.** Run `git status`; if the draw-avoidance and `play` changes (`policy.py`, `rules.py`, `play.py`, `cli.py`, `tests/test_policy.py`, `tests/test_play.py`) are uncommitted, run the suite, then commit them as `feat: bot never draws; play command`.
- [ ] **Step 2: Write failing tests** in `tests/test_split.py`: `test_three_way_split_is_ordered_and_disjoint` (build 40 games with increasing `end_time`; assert all train < all val < all test by `end_time`, no overlap, `test` matches `split_by_date` output, sizes 29/4/6 for 0.15/0.10 on 40 games), `test_three_way_split_needs_three_games`.
- [ ] **Step 3: Run** `.venv/bin/pytest -q tests/test_split.py`. Expected: the two new tests FAIL (function missing).
- [ ] **Step 4: Implement `three_way_split`** in `src/chessbot/split.py` reusing `split_by_date` twice.
- [ ] **Step 5: Run** the split tests. Expected: all PASS.
- [ ] **Step 6: Write failing slow tests** in `tests/test_maia.py`: `test_checkpoint_argument_loads_that_file` (save the 5M model's `state_dict` with one weight tensor scaled by 1.5 to `tmp_path/"ft.pt"` as `{"model_state_dict": ...}`, build `MaiaPredictor("maia3-5m", checkpoint=that)`, assert its probabilities on the start position differ from the base predictor's), `test_missing_checkpoint_raises`, and `test_history_frames_matches_predictor_history` (for a 12-move game, `history_frames(board, 8)` equals `predictor._history(board)` frame by frame).
- [ ] **Step 7: Run** `.venv/bin/pytest -q -m "slow or not slow" tests/test_maia.py`. Expected: the new tests FAIL.
- [ ] **Step 8: Implement** the `history_frames` extraction and the `checkpoint` argument in `src/chessbot/maia.py`.
- [ ] **Step 9: Run** the whole suite. Expected: all PASS, no regressions.
- [ ] **Step 10: Commit** `feat: three-way split; MaiaPredictor accepts a checkpoint`.

---

### Task 2: Training examples

**Files:**
- Create: `src/chessbot/examples.py`
- Test: `tests/test_examples.py` (fast) and one slow test in the same file

**Interfaces:**
- Consumes: `Game` (`moves`, `my_color`, `my_elo`, `opp_elo`, `time_class`), `history_frames`, maia3's `tokenize_board`, `get_legal_moves_mask`, `mirror_move`, `get_all_possible_moves`.
- Produces:
  - `@dataclass(frozen=True) class Example` with fields `frames: np.ndarray` (shape `(8, 64)`, dtype int8, piece codes 0 = empty, 1-12 as maia3's token index + 1, model orientation, oldest first, short histories padded by repeating the earliest frame), `self_elo: int`, `oppo_elo: int`, `target: int` (vocabulary index of the played move, mirrored for Black), `legal: np.ndarray` (int16 vocabulary indices of all legal moves, mirrored for Black), `weight: float`.
  - `build_examples(games: Iterable[Game], move_index: dict[str, int], bullet_weight: float = 0.5, history: int = 8) -> list[Example]`: one example per move Justin played; weight = `bullet_weight` for `time_class == "bullet"`, else 1.0.
  - `expand_frames(frames: np.ndarray) -> torch.Tensor`: `(B, 8, 64)` int8 to `(B, 64, 96)` float, equal to `get_historical_tokens(...)[:, :96]` for the same position.
  - `collate(examples: Sequence[Example], vocab_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]` returning `(tokens (B,64,96), self_elo (B,) long, oppo_elo (B,) long, target (B,) long, legal_mask (B,V) bool, weight (B,) float)`.
  - `save_examples(examples, path: Path)` / `load_examples(path: Path) -> list[Example]` using one compressed `.npz` (stacked arrays, legal indices stored flat with offsets).

- [ ] **Step 1: Write failing tests** (fast, use `get_all_possible_moves()` for the index dict, no model): `test_one_example_per_move_by_justin_only` (a 6-ply game as White gives 3 examples; the same game as Black gives 3, with targets taken from plies 2, 4, 6), `test_black_target_and_legal_indices_are_mirrored` (Black's first reply `e7e5`: target index equals `move_index[mirror_move("e7e5")]` and is in `legal`; `legal` has 20 entries), `test_promotion_target_is_valid_and_legal` (a hand-built game that ends with Justin promoting; target in `legal`, index < vocab size), `test_single_legal_move_position_still_yields_an_example`, `test_bullet_moves_are_down_weighted`, `test_expand_frames_matches_get_historical_tokens` (compare against `get_historical_tokens` on `history_frames` of a mid-game board AND of the start position, where the history is padded), `test_save_load_round_trip`, `test_collate_masks_exactly_the_legal_moves`.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_examples.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `src/chessbot/examples.py`.** Replay each game keeping a `deque(maxlen=history)` of `tokenize_board` frames; at each of Justin's moves compress the frames to int8 codes, compute target and legal indices from the board (mirroring for Black), then push the move. One approach for compression: `codes = (t.argmax(dim=1) + 1) * t.any(dim=1)`; expansion via `F.one_hot(codes.long(), 13)[..., 1:]`.
- [ ] **Step 4: Run** the fast tests. Expected: PASS.
- [ ] **Step 5: Write the slow test** `test_training_encoding_gives_the_same_probabilities_as_the_predictor` (5M model, first 20 plies of a real-style game: for each Justin move, softmax of the model on `collate`'d tokens, masked to `legal`, equals `MaiaPredictor.move_probs` for the same board within 1e-5).
- [ ] **Step 6: Run** `.venv/bin/pytest -q -m "slow or not slow" tests/test_examples.py`. Expected: PASS (if it fails, the encoding differs from inference; fix the encoding, not the test).
- [ ] **Step 7: Commit** `feat: training examples from Justin's moves`.

---

### Task 3: Training loop

**Files:**
- Create: `src/chessbot/finetune.py`
- Test: `tests/test_finetune.py` (fast, stub model) and one slow test

**Interfaces:**
- Consumes: `Example`, `collate` from Task 2.
- Produces:
  - `masked_loss(logits: Tensor, legal_mask: Tensor, target: Tensor, weight: Tensor, base_logits: Tensor | None = None, kl_weight: float = 0.0) -> Tensor`: weighted mean cross-entropy over legal moves only (illegal logits filled with `-inf` before the softmax); when `base_logits` is given, adds `kl_weight * KL(base || current)` over legal moves, same weighting.
  - `@dataclass class TrainResult: best_state: dict[str, Tensor]; best_epoch: int; history: list[dict]` where each history row has `epoch`, `train_loss`, `val_loss`, `val_top1`.
  - `train(model, train_examples, val_examples, *, vocab_size: int, epochs: int = 3, lr: float = 1e-5, batch_size: int = 64, kl_weight: float = 0.0, freeze_layers: int = 0, patience: int = 1, device: str = "cpu", seed: int = 0, log: Callable[[str], None] = print) -> TrainResult`. Model contract: `model(tokens, self_elo, oppo_elo)[0]` gives logits. `freeze_layers=n` sets `requires_grad=False` on `model.transformer.layers[:n]`. When `kl_weight > 0`, a frozen deep copy of the starting model supplies `base_logits`. Validation uses the same loss without the KL term and `weight` of 1 for every example. Stops after `patience` epochs without a better validation loss. Restores nothing into the model itself; `best_state` is a CPU copy of the best epoch's `state_dict`.
  - `save_checkpoint(state: dict, path: Path) -> None`: writes `{"model_state_dict": state}`, creating parent folders, and raises `FileExistsError` if `path` exists.

- [ ] **Step 1: Write failing tests** in `tests/test_finetune.py`:
  - `test_masked_loss_ignores_illegal_moves`: logits with a huge value on an illegal index leave the loss unchanged.
  - `test_masked_loss_matches_hand_computation`: two legal moves, logits `[0, ln 3]`, target the first, loss = `-ln(0.25)`.
  - `test_weights_scale_the_loss`: weight 0.5 on one of two equal examples gives the weighted mean.
  - `test_kl_term_is_zero_when_model_equals_base_and_positive_otherwise`.
  - A small stub `nn.Module` with `forward(tokens, self_elo, oppo_elo) -> (logits, None, None)` (an `nn.Linear` over flattened tokens plus an elo term) and synthetic examples where the target is a learnable function of the input: `test_training_reduces_validation_loss`, `test_best_state_is_the_best_epoch_not_the_last` (load `best_state` into the stub and recompute validation loss; it equals the minimum `val_loss` in `history` within 1e-6; use a large `lr` and `epochs=6` so later epochs get worse), `test_patience_stops_early`, `test_freeze_layers_leaves_frozen_weights_unchanged` (stub exposes `transformer.layers`), `test_train_does_not_modify_the_model_passed_in_beyond_normal_updates` is NOT required; skip it.
  - `test_save_checkpoint_refuses_to_overwrite` and `test_checkpoint_format_matches_maia3_loader` (the file has key `model_state_dict`).
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_finetune.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `src/chessbot/finetune.py`.** AdamW, fp32, shuffled batches with a seeded generator, `model.train()` for training and `model.eval()` plus `no_grad` for validation.
- [ ] **Step 4: Run** the fast tests. Expected: PASS.
- [ ] **Step 5: Write the slow test** `test_five_m_model_overfits_a_handful_of_positions` (5M model, 16 examples built from one real-style game, `epochs=30`, `lr=1e-4`, validation = the same 16: final training loss is below the starting loss by at least 30%). This is the spike from the design: it proves gradients flow through the real model.
- [ ] **Step 6: Run** `.venv/bin/pytest -q -m "slow or not slow" tests/test_finetune.py`. Expected: PASS. If gradients do not flow (loss unchanged), stop and report: the plan's premise is broken.
- [ ] **Step 7: Commit** `feat: fine-tune training loop`.

---

### Task 4: Commands and checkpoint selection

**Files:**
- Modify: `src/chessbot/cli.py`
- Create: `src/chessbot/compare.py`
- Test: `tests/test_cli.py`, `tests/test_compare.py`, slow checkpoint-compat test in `tests/test_finetune.py`

**Interfaces:**
- Consumes: `three_way_split`, `build_examples`, `save_examples`, `load_examples`, `train`, `save_checkpoint`, `MaiaPredictor(..., checkpoint=...)`.
- Produces:
  - `common()` in `cli.py` gains `--checkpoint` (a `Path`, default `None`) passed to every `MaiaPredictor(...)` construction in `evaluate`, `calibrate`, `tune-temperature`, `book-depth`, `play`. `evaluate` also gains `--calibration` (default `data/calibration.json`) replacing the hardcoded path.
  - `export-examples`: `--raw`, `--username`, `--out-dir` (default `data/finetune`); writes `train.npz` and `val.npz` built with `three_way_split`; test games are never written. Prints the three sizes.
  - `finetune`: `--examples` (dir with `train.npz`, `val.npz`), `--model` (default `maia3-23m`), `--out` (checkpoint path, required), `--epochs`, `--lr`, `--batch-size`, `--kl-weight`, `--freeze-layers`, `--patience`, `--device` (default `cpu`; `cuda` or `mps` allowed), `--max-examples` (optional cap for smoke runs). Loads the base model through `MaiaPredictor` to get the model and vocabulary, trains, prints the history table, saves the best state with `save_checkpoint`.
  - `compare.py`: `top_moves(predictor, board, elo, n=5) -> list[tuple[str, float]]` (SAN, probability, descending) and `format_comparison(board, columns: dict[str, list[tuple[str, float]]]) -> str` (one block, a column per model name).
  - `compare-position`: `--fen`, `--elo` (default calibrated), `--checkpoint` (fine-tuned file); prints base and fine-tuned top moves side by side.

- [ ] **Step 1: Write failing tests.**
  - `tests/test_cli.py::test_export_examples_never_writes_test_games` and `::test_export_validation_games_are_newer_than_training_games` (use the existing test_cli raw-data fixture style; check by example counts per game and by `end_time` ordering recorded through monkeypatching `build_examples`, or build the raw dir so game lengths identify the split).
  - `tests/test_cli.py::test_finetune_refuses_to_overwrite_existing_checkpoint` (stub the model loader; exit with an error before training).
  - `tests/test_compare.py::test_top_moves_sorted_and_in_san` and `::test_format_comparison_has_a_column_per_model` (stub predictors with fixed `move_probs`).
  - Slow, in `tests/test_finetune.py`: `test_saved_checkpoint_reloads_into_predictor_with_same_probabilities` (train 1 epoch on a few examples with the 5M model, `save_checkpoint`, `MaiaPredictor("maia3-5m", checkpoint=path)` probabilities equal the in-memory fine-tuned model's masked softmax within 1e-5).
- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** `compare.py` and the CLI changes (including `--checkpoint` plumbing and the new subcommands).
- [ ] **Step 4: Run** the whole suite with `-m "slow or not slow"`. Expected: all PASS.
- [ ] **Step 5: Smoke-run locally** with the 5M model on a small cap: `.venv/bin/python -m chessbot export-examples`, then `.venv/bin/python -m chessbot finetune --examples data/finetune --model maia3-5m --max-examples 2000 --epochs 1 --out data/smoke-5m.pt`, then `.venv/bin/python -m chessbot evaluate --policy maia --model maia3-5m --checkpoint data/smoke-5m.pt --max-games 20`. Expected: each command exits 0 and the evaluate table prints. Delete `data/smoke-5m.pt` afterwards.
- [ ] **Step 6: Commit** `feat: export-examples, finetune, compare-position, --checkpoint`.

---

### Task 5: Colab runbook

**Files:**
- Create: `docs/finetune-colab.md`

**Interfaces:**
- Consumes: the commands from Task 4.
- Produces: a runbook Justin can follow without Claude.

- [ ] **Step 1: Write `docs/finetune-colab.md`:** how to run `export-examples` locally and zip `data/finetune`; how to open a Colab notebook with a GPU runtime, install `torch`, `python-chess` and the repo's pinned `maia3` from GitHub (copy the exact install line from `pyproject.toml`), unzip the repo and the examples, run `python -m chessbot finetune --examples ... --model maia3-23m --device cuda --out finetuned-23m.pt` with starting values `--lr 1e-5 --epochs 3 --patience 1` and the alternatives to try if validation loss rises early (`--kl-weight 0.1`, `--freeze-layers 4`), and how to download `finetuned-23m.pt` into `data/` (never into the Hugging Face cache).
- [ ] **Step 2: Verify** every command in the runbook against `python -m chessbot <cmd> --help` locally. Expected: the flags exist.
- [ ] **Step 3: Commit** `docs: Colab fine-tune runbook`.

**Stop here until Justin has a trained checkpoint.** Training is done by Justin on Colab; the next task needs the file.

---

### Task 6: Evaluate base against fine-tuned

**Files:**
- Create: `docs/results/finetune-<date>.md`

**Interfaces:**
- Consumes: `data/finetuned-23m.pt` from Justin's Colab run; all Task 4 commands.

- [ ] **Step 1: Accuracy.** Run `evaluate` on the full test split for base Maia, the current harness, and the fine-tuned model with and without the table (`--policy maia --checkpoint data/finetuned-23m.pt`, `--policy harness --checkpoint ...`), each at the calibrated Elo and temperature 1.0. Expected: tables for top-1, top-3, mean log-probability by phase and time class.
- [ ] **Step 2: Recalibrate.** Run `calibrate --checkpoint data/finetuned-23m.pt --out data/calibration-ft.json`, then `tune-temperature --checkpoint data/finetuned-23m.pt --calibration data/calibration-ft.json --out data/temperature-ft.json` with at least 100 games per temperature and the grid `0.8 0.9 1.0 1.1 1.2`. Expected: a chosen Elo and temperature whose self-play average centipawn loss is within about 10% of Justin's 68.8.
- [ ] **Step 3: Spot checks.** Run `compare-position --fen "r1bq1rk1/ppp1ppbp/2np1np1/8/2PPP3/2N1BN2/PP2BPPP/R2QK2R b KQ - 0 8" --elo 1700 --checkpoint data/finetuned-23m.pt`. Expected: a side-by-side of top moves; record whether e5 gained probability.
- [ ] **Step 4: Write `docs/results/finetune-<date>.md`** with the tables, the train-versus-validation loss gap, the chosen settings, and a verdict against the design's success measures (middlegame and endgame top-1 at least +2 points over 57.8%, better mean log-probability, opening not worse, error profile within about 10%). Say plainly if the bar was missed.
- [ ] **Step 5: Play check.** Give Justin the commands to play both versions (`play --color white`, then the same with `--checkpoint data/finetuned-23m.pt --calibration data/calibration-ft.json --tuned data/temperature-ft.json`), and record which one Justin says feels more like them.
- [ ] **Step 6: Run** the whole suite. Expected: all PASS. **Commit** `docs: fine-tune results`.
