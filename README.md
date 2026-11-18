# A chess bot that plays like me

I fine-tuned Maia 3 on my Chess.com games to make a bot that plays like me: my openings, my habits, and my mistakes. It samples moves from a personal position table when it has seen the position often enough, then uses the fine-tuned model everywhere else.

There is no search. The goal is to predict my moves and match my playing style.

## Results

The original dataset has $10{,}040$ standard-chess games for `Justin0921`. The newest $1{,}506$ games are held out for testing. The comparison below uses the same $52{,}965$ test positions for every model, excluding flagged time-scramble moves, with Elo input $1700$ and temperature $1$. It evaluates the model alone, without the position table.

| Model | Top-1 move prediction | Opening | Middlegame | Endgame |
| --- | --- | --- | --- | --- |
| Base Maia 3 | $59.1\%$ | $61.0\%$ | $58.1\%$ | $59.4\%$ |
| Fine-tune 1 | $63.8\%$ | $72.4\%$ | $60.4\%$ | $60.7\%$ |
| Fine-tune 2 | $63.9\%$ | $72.2\%$ | $60.5\%$ | $61.1\%$ |

Fine-tune 2 excludes bullet moves made with at most $5$ seconds remaining after the move that lost at least $200$ centipawns. Its move prediction is essentially tied with fine-tune 1. I picked it because cleaner training data fits the clock-free bot I wanted.

I also played blinded trials and judged whether the bot played like me. In the final focused set, base passed $0/7$, fine-tune 1 passed $1/6$, and fine-tune 2 passed $3/7$. I was the only judge, and the sample is small. These are subjective play trials, not a controlled strength test. The local log does not preserve the moves or trial lengths. See [the blind-test report](docs/results/blind-and-freeze-2026-11-15.md), including the earlier trials reported from the preceding session.

The selected bot's self-play average centipawn loss was $66.2$, compared with $66.8$ in a sample of my scramble-filtered training games. This is a strength-profile proxy, not a measured playing rating. Full experiment details: [baseline](docs/results/baseline-2026-11-02.md), [fine-tune 1](docs/results/finetune-2026-11-03.md), [fine-tune 2](docs/results/finetune2-2026-11-05.md).

## Play on the website

[**Play Justin**](https://justee21.github.io/chess-bot/web/)

Click Play Justin to reveal the board, then choose White or Black. The model runs on your device, with no clock. First use downloads roughly $106\,\mathrm{MB}$ of model and runtime files; later games can use your browser’s cache. [Browser implementation, parity checks, and build instructions](docs/browser.md).

## Run the frozen bot

The model runs on CPU. Create the environment in the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[inference,dev]'
python scripts/fetch_artifacts.py
```

The checkpoint and position table are [GitHub Release assets](https://github.com/Justee21/chess-bot/releases/tag/v0.1.0-frozen), outside Git history. The download script checks both SHA-256 hashes against [config/final.json](config/final.json). It refuses to replace different local files.

To play in the terminal, copy the saved calibration files and select the released checkpoint:

```bash
cp config/runtime/calibration-ft-clean.json data/
cp config/runtime/temperature-ft-clean.json data/
python -m chessbot play \
  --checkpoint data/finetuned-23m-clean.pt \
  --calibration data/calibration-ft-clean.json \
  --tuned data/temperature-ft-clean.json
```

The UCI entry point loads all settings directly from the frozen config:

```bash
python -m chessbot.uci --config config/final.json
```

| Setting | Frozen value |
| --- | --- |
| Checkpoint | `data/finetuned-23m-clean.pt` |
| Maia 3 revision | `1e13597c42d4858b7cfd7cfdae01e297263364b2` |
| Elo conditioning input | $1500$ for both players |
| Temperature | $1.2$ |
| Relative rare-move cutoff | $0.05$ |
| Personal table | `data/table.json`, minimum total position count $3$ |
| Search / resignation / draw offers or acceptance | Disabled |

The cutoff is relative to the most likely playable move, not an absolute probability threshold. Temperature affects the model distribution; table hits use empirical frequencies. The existing policy avoids immediately drawing moves when alternatives exist, but forced and automatic draws can still happen.

## Reproduce the experiments

Base model: [Maia 3 / Chessformer](https://github.com/CSSLab/maia3), the `maia3-23m` architecture at commit `1e13597c42d4858b7cfd7cfdae01e297263364b2`. The inference extra pins that revision.

```bash
python -m chessbot download --username Justin0921
python -m chessbot build-table
python -m chessbot mark-scrambles
python -m chessbot export-examples --out-dir data/finetune-clean --scrambles data/scrambles.json
python -m chessbot finetune --examples data/finetune-clean --device cuda \
  --lr 1e-5 --epochs 3 --patience 1 --out data/retrained-23m-clean.pt
```

Stockfish must be installed for `mark-scrambles`, strength profiles, and temperature tuning. Fine-tuning ran on a Colab T4; see [the cleaned-data runbook](docs/finetune-colab-clean.md). The best checkpoint was epoch $1$; the second epoch worsened validation loss. Bullet training examples have weight $0.5$; other time controls have weight $1$.

Splits are chronological by whole game: the newest $15\%$ is test. Of the remaining $8{,}534$ games, the newest $10\%$ ($853$ games) is validation and the oldest $7{,}681$ games is fine-tuning data. The position table and calibration use the pre-test portion; the fine-tune uses only its training slice. The test games are not exported for training.

For the recorded model-only comparison:

```bash
python -m chessbot evaluate --policy maia --elo 1700 --temperature 1 \
  --checkpoint data/finetuned-23m-clean.pt --scrambles data/scrambles.json
```

Downloading the account today includes later games and can change the split. [The dataset manifest](docs/dataset-manifest.json) records the original archive hashes, game counts, and date boundaries without publishing opponents or games. Exact historical reruns require the original private archive snapshot. Training also has GPU-dependent variation; the released weights and table are the exact published artifacts.

## Tests

```bash
python -m pytest -q
```

Tests marked `slow` require model weights and/or Stockfish and are excluded by default. The UCI adapter is also tested with real model inference and `python-chess` engine communication before release.

## Data and license

Raw Chess.com archives, training examples, local play logs, and access tokens are excluded from Git and releases. The repository contains synthetic game fixtures for tests. The released table contains board positions and move counts, so my repertoire is public; it contains no player names or game URLs.

Code and released fine-tuned weights are distributed under [AGPL-3.0-or-later](LICENSE). Maia 3 is by the University of Toronto CSSLab; its original code and license remain available at the [pinned revision](https://github.com/CSSLab/maia3/tree/1e13597c42d4858b7cfd7cfdae01e297263364b2). See [attribution](NOTICE.md).
