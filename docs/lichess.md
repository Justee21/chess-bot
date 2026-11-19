# Run the frozen bot on Lichess

## Prepare the model

Follow the install and artifact-download steps in the [README](../README.md). The UCI wrapper always reads `config/final.json`; it does not expose Elo, temperature, or cutoff options to the Lichess bridge.

## Prepare the bridge

Use Python 3.11 or later. From this repository, with `.venv` activated:

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git lichess-bot
git -C lichess-bot checkout 06f4f10b5a5a70549b9fc5f8f759a532723b5dab
python -m pip install -r lichess-bot/requirements.txt
python scripts/configure_lichess.py --bridge lichess-bot
```

The generator refuses to overwrite an existing bridge config. It creates `lichess-bot/config.yml` with absolute engine paths and an environment-token placeholder. That directory is ignored by Git. Run the generator again on a new host because its paths will differ.

Initial service configuration accepts standard casual blitz/rapid with at least 180 seconds base time, at most 1800 seconds base and 30 seconds increment. It caps concurrency at one game. Matchmaking and takebacks are disabled. External books, cloud analysis, tablebases, pondering, resignation, and draw offers/acceptance are disabled. All chat greetings are blank.

The policy still samples moves and avoids immediate draws as it did during the play trials. Forced and automatic draws remain possible. It does not use the game's actual ratings or clock as model inputs. No artificial delay is added.

## Account and token

Use a dedicated fresh account, never your human playing account. Follow the [official bot-account instructions](https://lichess.org/api#tag/Bot) and create a personal access token on that account with the `bot:play` scope.

The upgrade to a BOT account is irreversible. Check the username before doing it. The bridge provides the upgrade command below; do not use it until the account identity has been checked.

Read the token without putting it into shell history:

```bash
export LICHESS_BOT_TOKEN="$(python -c 'import getpass; print(getpass.getpass("Lichess bot token: "))')"
```

Then, from the bridge directory, upgrade the dedicated account once:

```bash
cd lichess-bot
../.venv/bin/python lichess-bot.py -u
```

The upgrade command also starts the bot. For later launches, omit `-u`:

```bash
../.venv/bin/python lichess-bot.py
```

Before opening the bot to visitors, verify a complete casual game and a reconnect. Check that the profile links to the published source and checkpoint. Suggested factual profile text:

> Chess bot trained to imitate Justin Roberts's play. Maia 3 fine-tuned on his Chess.com moves, plus his personal position table. No search; samples moves with frozen settings. Currently accepts casual standard blitz and rapid. Source and model: https://github.com/Justee21/chess-bot

This repository does not create a Lichess account or store a token. No account conversion or live game is implied by installing the adapter.

## Local checks

```bash
python -m pytest -q
python -m chessbot.uci --config config/final.json
```

For a manual UCI check, enter `uci`, `isready`, `position startpos moves e2e4`, `go`, and `quit`. The responses include `uciok`, `readyok`, and one legal `bestmove`. Model diagnostics go to stderr. Artifact hash mismatches stop startup instead of silently falling back to base Maia.

Before release, also run actual model communication through `python-chess` and local games against Stockfish at skill 0. These establish integration and legal play, not a human strength estimate. Live authentication, account conversion, challenge acceptance, and reconnection still require separate online checks.

## Keeping it available

On a Mac, the bot is available while the bridge process runs and the machine is awake and connected. From the bridge directory, `caffeinate` can prevent idle sleep during a session:

```bash
caffeinate -i ../.venv/bin/python lichess-bot.py
```

For continuous availability, install the same pinned software on an always-on Linux host and run the bridge under a process supervisor. Keep the token in the host's secret/environment configuration. No hosting service or paid resources are created by this repository.

Once the account is live, link to its Lichess profile from the website and state the actual uptime arrangement. The initial casual-only configuration produces no rating. If rated play is enabled later, it is a service change; Maia's fixed 1500 conditioning input is not a measured Lichess rating.
