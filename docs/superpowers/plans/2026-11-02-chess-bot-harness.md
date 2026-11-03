# Chess Bot Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local harness that plays chess like Justin (opening/position table from his chess.com games, Maia 3 as fallback) plus the metrics that say how close it is.

**Architecture:** `games.py` ingests chess.com archives into `Game` records; `split.py` separates oldest/newest by date; `table.py` counts Justin's moves per position from training games only; `maia.py` wraps the Maia 3 checkpoint; `policy.py` combines table and Maia and samples; `evaluate.py`, `cpl.py`, `selfplay.py`, `calibrate.py` measure it. A small `cli.py` exposes each step.

**Tech Stack:** Python >= 3.10, python-chess, torch, numpy, pytest, `maia3` (installed from GitHub, AGPL-3.0), Stockfish (`brew install stockfish`).

**Spec:** `docs/superpowers/specs/2026-11-02-chess-bot-design.md`

## Global Constraints

- chess.com username `Justin0921`; standard chess only (`rules == "chess"`); all time classes kept; metrics also reported per time class.
- Split by date: oldest 85% of games train, newest 15% test; the test set is used only for final measurements.
- Table is keyed by board position, built from training games only, from Justin's own moves only; a position is used only if it was seen at least 3 times.
- Fallback is Maia 3, no search. Moves are sampled from a distribution, never argmax, with one fixed temperature tuned once against centipawn loss. The bot never resigns.
- Target strength is Justin's lifetime average; the Maia Elo input is calibrated (chess.com ratings are not Lichess-scale), using training games only.
- Raw PGNs and opponent usernames stay out of any public repo: `data/` is gitignored.
- Budget is $0. Maia 3 is AGPL-3.0, so the repo carries an AGPL-3.0 `LICENSE` and cites the Chessformer paper (Monroe et al., ICLR 2026).
- Out of scope here: fine-tuning (separate plan, only if this baseline shows a gap), hosting/ONNX, the LLM judge, the blind human test, a Lichess bot.

## Review Focus

1. **Test-set leakage:** the table, Elo calibration, and temperature tuning must never see test games.
2. **Only Justin's moves:** opponents' moves must not enter the table or the accuracy metrics, for either color.
3. **Black and promotions:** Maia mirrors the board for Black; moves must mirror back, and promotion moves (`a7a8q`) must survive the round trip.
4. **Position keys:** the same position reached by different move orders must share a key; different castling rights must not.
5. **Degenerate positions:** no-legal-move positions (mate, stalemate), single-legal-move positions, and games chess.com returns without a usable PGN.

---

### Task 1: Project scaffold and game ingestion

**Files:**
- Create: `pyproject.toml`, `.gitignore` (`data/`, `__pycache__/`, `.venv/`), `LICENSE` (copy of the AGPL-3.0 text from the `CSSLab/maia3` repo), `src/chessbot/__init__.py`, `src/chessbot/games.py`, `src/chessbot/cli.py`, `tests/test_games.py`, `tests/fixtures/archive_sample.json`
- Run `git init` in `chess-bot/`.

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Game:
      url: str; end_time: int; time_class: str   # "bullet"|"blitz"|"rapid"|"daily"
      my_color: chess.Color; my_elo: int; opp_elo: int
      moves: tuple[str, ...]                     # whole game, UCI
  def download_archives(username: str, out_dir: Path, delay: float = 0.3) -> int  # months written
  def load_games(raw_dir: Path, username: str = "Justin0921") -> list[Game]       # sorted by end_time
  ```
  `cli.py`: `main(argv)` with subcommands; this task adds `download --out data/raw`.

- [ ] **Step 1: Write failing tests** in `tests/test_games.py` using a fixture archive of four games (standard blitz as White, standard bullet as Black, a bughouse game with no `pgn`, a chess960 game):
  `test_load_games_keeps_only_standard_chess` (len == 2), `test_load_games_sorted_by_end_time`, `test_my_color_and_elos_follow_username` (case-insensitive match; Black game gives `my_elo` from the black side), `test_moves_are_uci_tuples`, `test_game_with_unparseable_pgn_is_skipped` (Review Focus 5).
- [ ] **Step 2: Run** `pytest tests/test_games.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `Game`, `load_games`, `download_archives`. Use `chess.pgn.read_game`; skip games with zero moves or `rules != "chess"`. `download_archives` GETs `https://api.chess.com/pub/player/{username}/games/archives`, then each month, writing `YYYY-MM.json`; use a generic User-Agent (no email).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Download real data.** Run `python -m chessbot download --out data/raw` then a one-liner printing `len(load_games(Path("data/raw")))`. Expected: ~71 files and about 10,000 games (earlier check: 10,045 standard).
- [ ] **Step 6: Commit** `feat: scaffold project and chess.com ingestion`.

### Task 2: Date split

**Files:** Create `src/chessbot/split.py`, `tests/test_split.py`

**Interfaces:**
- Consumes: `Game` (Task 1).
- Produces: `def split_by_date(games: Sequence[Game], test_fraction: float = 0.15) -> tuple[list[Game], list[Game]]`. Test = newest `max(1, round(n * test_fraction))` games.

- [ ] **Step 1: Write failing tests:** `test_test_set_is_newest` (shuffled 20 games: `max(train.end_time) <= min(test.end_time)`, test size 3), `test_partition_is_complete_and_disjoint`, `test_fewer_than_two_games_raises_valueerror`.
- [ ] **Step 2: Run** `pytest tests/test_split.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `split_by_date`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: date-based train/test split`.

### Task 3: Move table

**Files:** Create `src/chessbot/table.py`, `tests/test_table.py`

**Interfaces:**
- Consumes: `Game`.
- Produces:
  ```python
  def position_key(board: chess.Board) -> str          # board.epd(): placement, side, castling, legal-ep only; no move clocks
  class MoveTable:
      @classmethod
      def build(cls, games: Iterable[Game]) -> "MoveTable"
      def counts(self, board: chess.Board) -> dict[str, int]          # UCI -> count; {} if unseen
      def lookup(self, board: chess.Board, min_count: int = 3) -> dict[str, float] | None
      def save(self, path: Path) -> None
      @classmethod
      def load(cls, path: Path) -> "MoveTable"
  ```
  `lookup` returns a normalized distribution when the position's total count >= `min_count`, else `None`. Only positions where `board.turn == game.my_color` are counted.

- [ ] **Step 1: Write failing tests:** `test_three_games_same_first_move_gives_certainty` (Justin as White plays `e2e4` from the start in 3 games: `lookup(start) == {"e2e4": 1.0}`), `test_two_games_below_min_count_returns_none`, `test_opponent_moves_not_counted` (Justin as White: position after 1.e4 has `counts == {}`) (Review Focus 2), `test_black_moves_counted_when_justin_is_black`, `test_transposition_shares_key` and `test_halfmove_clock_ignored` and `test_castling_rights_change_key` (Review Focus 4), `test_save_load_roundtrip`.
- [ ] **Step 2: Run** `pytest tests/test_table.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `position_key` and `MoveTable`; store `dict[str, Counter[str]]`, JSON on disk.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: CLI.** Add `build-table --raw data/raw --out data/table.json` (loads games, splits, builds from train only, prints train/test sizes and number of positions with count >= 3). Run it. Expected: it prints counts and writes `data/table.json`.
- [ ] **Step 6: Commit** `feat: position-keyed move table`.

### Task 4: Maia 3 predictor

**Files:** Create `src/chessbot/maia.py`, `tests/test_maia.py` (mark every test `@pytest.mark.slow`; register the marker in `pyproject.toml`)

**Interfaces:**
- Produces:
  ```python
  class MaiaPredictor:
      def __init__(self, model: str = "maia3-23m", device: str = "cpu")
      def move_probs(self, board: chess.Board, self_elo: int, oppo_elo: int) -> dict[str, float]
  ```
  Keys are UCI moves in the real board's orientation, only legal moves, summing to 1. History comes from `board.move_stack`.

- [ ] **Step 1: Install** (ask Justin first; these download code and weights): `pip install "maia3 @ git+https://github.com/CSSLab/maia3.git" python-chess`, then cache the 5M (~21 MB) and 23M (~92 MB) checkpoints from Hugging Face via `maia3-cache --model maia3-5m` / `maia3-23m`.
- [ ] **Step 2: Write failing tests** (use `maia3-5m` for speed): `test_start_position_probs_sum_to_one_over_legal_moves`, `test_start_position_prefers_common_openings` (`e2e4` or `d2d4` in top 3 at Elo 1500), `test_black_to_move_returns_black_moves` (after 1.e4, keys are legal Black moves and `e7e5` or `c7c5` is in the top 3) (Review Focus 3), `test_promotion_moves_survive_mirroring` (`8/P7/8/8/8/8/8/k6K w - - 0 1` includes `a7a8q`, `a7a8n`; and a Black promotion position) (Review Focus 3), `test_single_legal_move_gets_probability_one` (Review Focus 5), `test_elo_changes_distribution`.
- [ ] **Step 3: Run** `pytest tests/test_maia.py -v -m slow`. Expected: FAIL.
- [ ] **Step 4: Implement** `MaiaPredictor` on top of the installed package's model loader, `tokenize_board`, `get_historical_tokens`, `get_legal_moves_mask`, and move dictionary (read `maia3/uci.py` for how it builds history and mirrors Black). Softmax over legal-masked move logits.
- [ ] **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit** `feat: Maia 3 move predictor`.

### Task 5: Combined policy

**Files:** Create `src/chessbot/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Consumes: `MoveTable.lookup`, `MaiaPredictor.move_probs`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Choice: move: chess.Move; source: str      # "table" | "maia"
  class HarnessPolicy:
      def __init__(self, table: MoveTable | None, maia: MaiaPredictor, elo: int,
                   temperature: float, min_count: int = 3)
      def distribution(self, board: chess.Board) -> tuple[dict[str, float], str]
      def choose(self, board: chess.Board, rng: random.Random) -> Choice
  ```
  Temperature applies only to Maia's distribution (`p ** (1/T)`, renormalized); table frequencies are used as-is. `elo` is passed as both self and opponent Elo. There is no resign method.

- [ ] **Step 1: Write failing tests** with a stub Maia: `test_table_hit_uses_table`, `test_table_miss_falls_back_to_maia`, `test_no_table_uses_maia`, `test_temperature_one_leaves_distribution_unchanged`, `test_lower_temperature_sharpens_and_still_sums_to_one`, `test_choose_samples_not_argmax` (two moves at 0.5 each: over 500 seeded draws both appear), `test_choose_with_no_legal_moves_raises_valueerror` (Review Focus 5).
- [ ] **Step 2: Run** `pytest tests/test_policy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `HarnessPolicy`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: table-then-Maia sampling policy`.

### Task 6: Move-match evaluation

**Files:** Create `src/chessbot/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Game`, `HarnessPolicy.distribution`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class MoveResult: top1: bool; top3: bool; logprob: float; phase: str; time_class: str; source: str
  def classify_phase(board: chess.Board) -> str   # "opening" if fullmove_number <= 10; else "endgame" if <= 10 non-king pieces; else "middlegame"
  def justin_positions(games: Iterable[Game]) -> Iterator[tuple[Game, chess.Board, str]]  # board before Justin's move, UCI played
  def evaluate(dist_fn: Callable[[chess.Board], tuple[dict[str, float], str]],
               games: Iterable[Game], max_games: int | None = None) -> list[MoveResult]
  def summarize(results: Sequence[MoveResult]) -> dict[str, dict[str, float]]  # keys "all", "phase:opening", "time:bullet", "source:table"...; values n, top1, top3, mean_logprob
  ```
  `logprob` is `log(max(p, 1e-6))` for the move Justin played.

- [ ] **Step 1: Write failing tests:** `test_justin_positions_yields_only_justins_turns` (both colors; count matches) (Review Focus 2), `test_evaluate_top1_top3_and_logprob` (stub dist `{actual: 0.6, other: 0.4}`), `test_zero_probability_move_is_floored`, `test_summarize_groups_by_phase_time_and_source`, `test_classify_phase_cases`.
- [ ] **Step 2: Run** `pytest tests/test_evaluate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: CLI.** Add `evaluate --policy maia|harness --elo N --temperature T [--max-games N]`, run on the **test** split, printing `summarize`. Run both policies on a small `--max-games 100` sample. Expected: two summaries print; this also times Maia on CPU for planning larger runs. (Final full-test-set numbers are produced in Task 10.)
- [ ] **Step 6: Commit** `feat: move-match evaluation`.

### Task 7: Elo calibration

**Files:** Create `src/chessbot/calibrate.py`, `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `MaiaPredictor`, `justin_positions`, `evaluate`.
- Produces: `def calibrate_elo(maia, train_games: Sequence[Game], candidates: Sequence[int] = range(800, 2201, 100), sample_games: int = 150, seed: int = 0) -> tuple[int, dict[int, float]]` — best Elo by mean log-probability of Justin's moves on a random sample of the games passed in, with scores per candidate.

- [ ] **Step 1: Write failing tests:** `test_picks_elo_with_best_mean_logprob` (stub predictor whose accuracy peaks at 1400), `test_only_uses_games_passed_in` (stub records every board it sees; none from a held-back test game) (Review Focus 1), `test_sampling_is_seeded`.
- [ ] **Step 2: Run** `pytest tests/test_calibrate.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: CLI.** Add `calibrate` (train split only; saves the chosen Elo and score table to `data/calibration.json`). Run it with the 23M model. Expected: a best Elo and a table that rises then falls; if the curve is flat or monotonic at an edge, widen `candidates` and rerun.
- [ ] **Step 6: Commit** `feat: Elo calibration against training games`.

### Task 8: Centipawn loss and error profile

**Files:** Create `src/chessbot/cpl.py`, `tests/test_cpl.py`

**Interfaces:**
- Produces:
  ```python
  def game_cpls(engine: chess.engine.SimpleEngine, moves: Sequence[str], color: chess.Color, depth: int = 12) -> list[float]
  def error_profile(cpls: Sequence[float]) -> dict[str, float]   # n, avg_cpl, p100, p200, p300 (share of moves losing more than that)
  ```
  Loss for a move = `max(0, best_eval_before − eval_after)` in centipawns from the mover's view, mate scores mapped with `mate_score=1000`, loss capped at 1000. Analyse each position once and reuse it as both "after" and the next "before".

- [ ] **Step 1: Install Stockfish** (ask Justin first): `brew install stockfish`.
- [ ] **Step 2: Write failing tests:** `test_error_profile_numbers` (hand-computed list), `test_game_cpls_zero_when_eval_holds` and `test_game_cpls_reports_drop` (stub engine returning scripted scores), `test_cpl_capped_at_1000`, and a `@pytest.mark.slow` real-engine test: `f2f3 e7e5 g2g4 d8h4` — White's `g2g4` has a large loss (>= 500).
- [ ] **Step 3: Run** `pytest tests/test_cpl.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run** including `-m slow`. Expected: PASS.
- [ ] **Step 6: CLI.** Add `justin-profile --games 300 --depth 12` (random sample of all games, Justin's moves only) printing `error_profile` overall and per time class. Run it. Expected: a profile table; this is the target the bot is tuned to.
- [ ] **Step 7: Commit** `feat: Stockfish centipawn-loss profile`.

### Task 9: Self-play and temperature tuning

**Files:** Create `src/chessbot/selfplay.py`, `tests/test_selfplay.py`

**Interfaces:**
- Consumes: `HarnessPolicy.choose`, `game_cpls`, `error_profile`.
- Produces:
  ```python
  def play_game(policy: HarnessPolicy, rng: random.Random, max_plies: int = 400) -> tuple[str, ...]  # UCI moves; ends at board.outcome(claim_draw=True) or max_plies
  def pick_temperature(profiles: dict[float, dict[str, float]], target: dict[str, float]) -> float  # T whose avg_cpl is closest to target["avg_cpl"]
  ```

- [ ] **Step 1: Write failing tests:** `test_play_game_ends_on_checkmate_or_draw` (stub policy forcing fool's mate), `test_play_game_respects_max_plies`, `test_play_game_is_seeded`, `test_pick_temperature_chooses_closest_avg_cpl`.
- [ ] **Step 2: Run** `pytest tests/test_selfplay.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The bot never resigns: games end only on chess termination or the ply cap.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: CLI.** Add `tune-temperature --temperatures 0.6 0.8 1.0 1.2 --games 40`: plays self-play games per temperature with the calibrated Elo and the train-only table, profiles both sides' moves with `game_cpls`, picks via `pick_temperature` against `justin-profile`, and saves to `data/temperature.json`. Run it. Expected: a per-temperature profile table and a chosen T.
- [ ] **Step 6: Commit** `feat: self-play and temperature tuning`.

### Task 10: Opening depth and end-to-end baseline

**Files:** Create `src/chessbot/opening.py`, `tests/test_opening.py`, `docs/results/baseline-2026-11-02.md` (written by the run)

**Interfaces:**
- Produces: `def book_exit_ply(table: MoveTable, moves: Sequence[str], my_color: chess.Color, min_count: int = 3) -> int` — the ply at which Justin's side first stands in a position the table does not cover (len(moves) if never).

- [ ] **Step 1: Write failing tests:** `test_exit_ply_when_table_runs_out`, `test_never_leaves_book_returns_game_length`, `test_only_justins_turns_count`.
- [ ] **Step 2: Run** `pytest tests/test_opening.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: End-to-end run on the full test split** with the calibrated Elo and tuned temperature: `evaluate --policy maia` and `evaluate --policy harness` (top-1, top-3, mean log-prob, by phase, time class, and source, including the share of positions the table covers), the harness error profile versus Justin's, and median book-exit ply for Justin's test games versus bot self-play. Write all numbers into `docs/results/baseline-2026-11-02.md`.
- [ ] **Step 6: Decide with Justin.** Put the harness-versus-Maia gap in the results file's last section. Fine-tuning gets its own plan only if that gap is worth closing.
- [ ] **Step 7: Commit** `feat: opening depth metric and baseline results`.
