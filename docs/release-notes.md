# Frozen fine-tune 2

First public release of Justin Roberts's personal chess imitation bot.

- Source includes the data pipeline, supervised fine-tuning, evaluation, personal position table policy, and UCI adapter for Lichess.
- The UCI adapter loads `config/final.json` and verifies the published checkpoint and table hashes.
- Model and table are release assets, not Git blobs. Download them with `python scripts/fetch_artifacts.py` after installing the package.
- Code, model, and table are distributed with the repository's AGPL-3.0-or-later license; see `NOTICE.md` for Maia 3 attribution.
- Raw training games, examples, play logs, and tokens are excluded.
- This release includes local engine checks; it does not imply that a Lichess account has been connected or that a bot is online.

## Validation

Standard suite: 133 tests passed, with 14 slow tests deselected.

The real checkpoint completed UCI initialization and 40 legal self-play plies. It also completed two local games against Stockfish at skill 0, one per color, with legal moves throughout. The generated deployment config was accepted by the pinned lichess-bot revision. These are local integration checks, not a strength estimate or a live Lichess test.

## Frozen policy

- Checkpoint: `finetuned-23m-clean.pt`
- SHA-256: `7068d98fa1bb3250081cdf506d35d45e46e91352b52a4802da0091e6b0ee8105`
- Table: `table.json`
- SHA-256: `55e8acf9149653626c4e3570f26f9bd5545f9aa48b3ab27cddaa8937847f1b51`
- Maia 3 revision: `1e13597c42d4858b7cfd7cfdae01e297263364b2`
- Elo input 1500, temperature 1.2, relative cutoff 0.05, minimum table position count 3.
- No search, resignation, or draw offers/acceptance. Forced and automatic draws remain possible.
