from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import chess
import chess.engine

from chessbot.cpl import error_profile, game_cpls
from chessbot.games import Game, download_archives, load_games
from chessbot.split import split_by_date
from chessbot.table import MoveTable


def _split(raw: Path, username: str) -> tuple[list[Game], list[Game]]:
    return split_by_date(load_games(raw, username))


def profile_sample(raw: Path, username: str, n: int, split: str = "train", seed: int = 0) -> list[Game]:
    """Random sample of Justin's games from one side of the date split only."""
    train, test = _split(raw, username)
    pool = train if split == "train" else test
    return random.Random(seed).sample(pool, min(n, len(pool)))


def _json_or(path: Path, key: str, default):
    return json.loads(path.read_text())[key] if path.exists() else default


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print(f"{'segment':<22}{'n':>8}{'top1':>8}{'top3':>8}{'mean_logp':>11}")
    for key in sorted(summary, key=lambda k: (k != "all", k)):
        s = summary[key]
        print(f"{key:<22}{int(s['n']):>8}{s['top1']:>8.3f}{s['top3']:>8.3f}{s['mean_logprob']:>11.3f}")


def _profile_games(engine, games: list[Game], depth: int) -> dict[str, dict[str, float]]:
    by_class: dict[str, list[float]] = {}
    for game in games:
        by_class.setdefault(game.time_class, []).extend(
            game_cpls(engine, game.moves, game.my_color, depth)
        )
    everything = [c for cpls in by_class.values() for c in cpls]
    out = {"all": error_profile(everything)}
    out.update({f"time:{k}": error_profile(v) for k, v in by_class.items()})
    return out


def _stockfish() -> chess.engine.SimpleEngine:
    path = shutil.which("stockfish")
    if not path:
        raise SystemExit("stockfish not found (brew install stockfish)")
    return chess.engine.SimpleEngine.popen_uci(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chessbot")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, table=False):
        p.add_argument("--raw", type=Path, default=Path("data/raw"))
        p.add_argument("--username", default="Justin0921")
        p.add_argument("--model", default="maia3-23m")
        if table:
            p.add_argument("--table", type=Path, default=Path("data/table.json"))

    download = sub.add_parser("download", help="download chess.com monthly archives")
    download.add_argument("--out", type=Path, default=Path("data/raw"))
    download.add_argument("--username", default="Justin0921")

    build = sub.add_parser("build-table", help="build the move table from training games")
    build.add_argument("--raw", type=Path, default=Path("data/raw"))
    build.add_argument("--out", type=Path, default=Path("data/table.json"))
    build.add_argument("--username", default="Justin0921")

    ev = sub.add_parser("evaluate", help="move-match accuracy on the TEST split")
    common(ev, table=True)
    ev.add_argument("--policy", choices=["maia", "harness"], required=True)
    ev.add_argument("--elo", type=int, default=None)
    ev.add_argument("--temperature", type=float, default=1.0)
    ev.add_argument("--max-games", type=int, default=None)

    cal = sub.add_parser("calibrate", help="pick the Maia Elo input using TRAIN games only")
    common(cal)
    cal.add_argument("--sample-games", type=int, default=150)
    cal.add_argument("--elo-range", type=int, nargs=3, default=[800, 2200, 100],
                     metavar=("START", "STOP", "STEP"))
    cal.add_argument("--out", type=Path, default=Path("data/calibration.json"))

    prof = sub.add_parser("justin-profile", help="Stockfish error profile of Justin's own games")
    prof.add_argument("--raw", type=Path, default=Path("data/raw"))
    prof.add_argument("--username", default="Justin0921")
    prof.add_argument("--split", choices=["train", "test"], default="train",
                      help="train feeds tuning; test is the final comparison number")
    prof.add_argument("--games", type=int, default=300)
    prof.add_argument("--depth", type=int, default=12)
    prof.add_argument("--out", type=Path, default=Path("data/justin_profile.json"))

    tune = sub.add_parser("tune-temperature", help="pick the sampling temperature by self-play")
    common(tune, table=True)
    tune.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    tune.add_argument("--games", type=int, default=40)
    tune.add_argument("--depth", type=int, default=12)
    tune.add_argument("--profile", type=Path, default=Path("data/justin_profile.json"))
    tune.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    tune.add_argument("--out", type=Path, default=Path("data/temperature.json"))

    depth = sub.add_parser("book-depth", help="how long Justin and the bot stay in the table")
    common(depth, table=True)
    depth.add_argument("--games", type=int, default=100)
    depth.add_argument("--temperature", type=float, default=None)
    depth.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    depth.add_argument("--tuned", type=Path, default=Path("data/temperature.json"))

    play = sub.add_parser("play", help="play the bot in the terminal")
    common(play, table=True)
    play.add_argument("--color", choices=["white", "black", "random"], default="white",
                      help="your color")
    play.add_argument("--temperature", type=float, default=None)
    play.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    play.add_argument("--tuned", type=Path, default=Path("data/temperature.json"))
    play.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "download":
        months = download_archives(args.username, args.out)
        print(f"wrote {months} monthly archives to {args.out}")

    elif args.command == "build-table":
        train, test = _split(args.raw, args.username)
        table = MoveTable.build(train)
        table.save(args.out)
        print(f"train games: {len(train)}, test games: {len(test)}")
        print(f"positions in table: {len(table)}, seen >= 3 times: {table.positions_with_min_count(3)}")
        print(f"wrote {args.out}")

    elif args.command == "evaluate":
        from chessbot.evaluate import evaluate, summarize
        from chessbot.maia import MaiaPredictor
        from chessbot.policy import HarnessPolicy

        _, test = _split(args.raw, args.username)
        elo = args.elo or _json_or(Path("data/calibration.json"), "elo", 1500)
        table = MoveTable.load(args.table) if args.policy == "harness" else None
        policy = HarnessPolicy(table, MaiaPredictor(args.model), elo, args.temperature)
        print(f"policy={args.policy} model={args.model} elo={elo} temperature={args.temperature}")
        _print_summary(summarize(evaluate(policy.distribution, test, args.max_games)))

    elif args.command == "calibrate":
        from chessbot.calibrate import calibrate_elo
        from chessbot.maia import MaiaPredictor

        train, _ = _split(args.raw, args.username)
        start, stop, step = args.elo_range
        best, scores = calibrate_elo(
            MaiaPredictor(args.model), train, range(start, stop + 1, step), args.sample_games
        )
        for elo, score in scores.items():
            print(f"{elo:>6}  mean_logprob {score:.4f}{'  <-- best' if elo == best else ''}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"elo": best, "scores": scores}))

    elif args.command == "justin-profile":
        sample = profile_sample(args.raw, args.username, args.games, args.split)
        with _stockfish() as engine:
            profile = _profile_games(engine, sample, args.depth)
        for key, p in profile.items():
            print(f"{key:<14} n={p['n']:<6} avg_cpl={p['avg_cpl']:.1f} "
                  f"p100={p['p100']:.3f} p200={p['p200']:.3f} p300={p['p300']:.3f}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(profile))

    elif args.command == "book-depth":
        import statistics

        from chessbot.maia import MaiaPredictor
        from chessbot.opening import book_exit_ply
        from chessbot.policy import HarnessPolicy
        from chessbot.selfplay import play_game

        _, test = _split(args.raw, args.username)
        table = MoveTable.load(args.table)
        justin = [book_exit_ply(table, g.moves, g.my_color) for g in test]
        elo = _json_or(args.calibration, "elo", 1500)
        temperature = args.temperature or _json_or(args.tuned, "temperature", 1.0)
        policy = HarnessPolicy(table, MaiaPredictor(args.model), elo, temperature)
        rng = random.Random(0)
        bot = []
        for _ in range(args.games):
            moves = play_game(policy, rng)
            bot.extend(book_exit_ply(table, moves, c) for c in (chess.WHITE, chess.BLACK))
        print(f"book-exit ply (Justin's turns), Justin test games: n={len(justin)} "
              f"median={statistics.median(justin)} mean={statistics.mean(justin):.1f}")
        print(f"book-exit ply, bot self-play (elo={elo}, T={temperature}): n={len(bot)} "
              f"median={statistics.median(bot)} mean={statistics.mean(bot):.1f}")

    elif args.command == "play":
        from chessbot.maia import MaiaPredictor
        from chessbot.play import play_game_vs_human
        from chessbot.policy import HarnessPolicy

        rng = random.Random(args.seed)
        human = {"white": chess.WHITE, "black": chess.BLACK}.get(args.color) \
            or rng.choice([chess.WHITE, chess.BLACK])
        elo = _json_or(args.calibration, "elo", 1500)
        temperature = args.temperature or _json_or(args.tuned, "temperature", 1.0)
        policy = HarnessPolicy(MoveTable.load(args.table), MaiaPredictor(args.model), elo, temperature)
        print(f"You are {'White' if human == chess.WHITE else 'Black'}. "
              f"Moves like e4, Nf3, O-O or e2e4. (elo={elo}, T={temperature})")
        result = play_game_vs_human(policy, human, rng)
        print(f"Result: {result}")

    elif args.command == "tune-temperature":
        from chessbot.maia import MaiaPredictor
        from chessbot.policy import HarnessPolicy
        from chessbot.selfplay import pick_temperature, play_game

        target = json.loads(args.profile.read_text())["all"]
        elo = _json_or(args.calibration, "elo", 1500)
        maia, table = MaiaPredictor(args.model), MoveTable.load(args.table)
        profiles = {}
        with _stockfish() as engine:
            for temperature in args.temperatures:
                policy = HarnessPolicy(table, maia, elo, temperature)
                rng = random.Random(0)
                cpls: list[float] = []
                for _ in range(args.games):
                    moves = play_game(policy, rng)
                    for color in (chess.WHITE, chess.BLACK):
                        cpls.extend(game_cpls(engine, moves, color, args.depth))
                profiles[temperature] = error_profile(cpls)
                print(f"T={temperature}: avg_cpl={profiles[temperature]['avg_cpl']:.1f} "
                      f"(target {target['avg_cpl']:.1f})", flush=True)
        best = pick_temperature(profiles, target)
        print(f"chosen temperature: {best}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"temperature": best,
                                        "profiles": {str(k): v for k, v in profiles.items()}}))
