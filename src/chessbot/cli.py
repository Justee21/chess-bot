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
from chessbot.scramble import find_scrambles, load_scrambles, save_scrambles, scramble_candidates
from chessbot.split import split_by_date, three_way_split
from chessbot.table import MoveTable


def _split(raw: Path, username: str) -> tuple[list[Game], list[Game]]:
    return split_by_date(load_games(raw, username))


def profile_sample(raw: Path, username: str, n: int, split: str = "train", seed: int = 0) -> list[Game]:
    """Random sample of Justin's games from one side of the date split only."""
    train, test = _split(raw, username)
    pool = train if split == "train" else test
    return random.Random(seed).sample(pool, min(n, len(pool)))


def export_examples(
    raw: Path, username: str, out_dir: Path, move_index: dict[str, int], bullet_weight: float = 0.5,
    skip: dict[str, set[int]] | None = None,
) -> tuple[int, int, int]:
    """Write training and validation examples; test games are counted but never written."""
    from chessbot.examples import build_examples, save_examples

    train, val, test = three_way_split(load_games(raw, username))
    save_examples(build_examples(train, move_index, bullet_weight, skip=skip), out_dir / "train.npz")
    save_examples(build_examples(val, move_index, bullet_weight, skip=skip), out_dir / "val.npz")
    return len(train), len(val), len(test)


def _json_or(path: Path, key: str, default):
    return json.loads(path.read_text())[key] if path.exists() else default


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print(f"{'segment':<22}{'n':>8}{'top1':>8}{'top3':>8}{'mean_logp':>11}")
    for key in sorted(summary, key=lambda k: (k != "all", k)):
        s = summary[key]
        print(f"{key:<22}{int(s['n']):>8}{s['top1']:>8.3f}{s['top3']:>8.3f}{s['mean_logprob']:>11.3f}")


def _profile_games(
    engine, games: list[Game], depth: int, skip: dict[str, set[int]] | None = None
) -> dict[str, dict[str, float]]:
    by_class: dict[str, list[float]] = {}
    for game in games:
        by_class.setdefault(game.time_class, []).extend(
            game_cpls(engine, game.moves, game.my_color, depth, skip=(skip or {}).get(game.url, ()))
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
        p.add_argument("--checkpoint", type=Path, default=None,
                       help="fine-tuned weights to use instead of the base model")
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
    ev.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    ev.add_argument("--temperature", type=float, default=1.0)
    ev.add_argument("--max-games", type=int, default=None)
    ev.add_argument("--scrambles", type=Path, default=None,
                    help="leave out the time-scramble moves listed in this file (see mark-scrambles)")

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
    prof.add_argument("--scrambles", type=Path, default=None,
                      help="leave out the time-scramble moves listed in this file")

    mark = sub.add_parser("mark-scrambles",
                          help="find bullet moves played in a time scramble that also lost a lot")
    mark.add_argument("--raw", type=Path, default=Path("data/raw"))
    mark.add_argument("--username", default="Justin0921")
    mark.add_argument("--max-seconds", type=float, default=5.0, help="clock left after the move")
    mark.add_argument("--loss", type=float, default=200.0, help="centipawns lost by the move")
    mark.add_argument("--depth", type=int, default=12)
    mark.add_argument("--out", type=Path, default=Path("data/scrambles.json"))

    tune = sub.add_parser("tune-temperature", help="pick the sampling temperature by self-play")
    common(tune, table=True)
    tune.add_argument("--temperatures", type=float, nargs="+", default=[0.6, 0.8, 1.0, 1.2])
    tune.add_argument("--min-p", type=float, default=0.05,
                      help="never sample a move under this fraction of the best move's probability")
    tune.add_argument("--games", type=int, default=40)
    tune.add_argument("--depth", type=int, default=12)
    tune.add_argument("--profile", type=Path, default=Path("data/justin_profile.json"))
    tune.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    tune.add_argument("--out", type=Path, default=Path("data/temperature.json"))

    depth = sub.add_parser("book-depth", help="how long Justin and the bot stay in the table")
    common(depth, table=True)
    depth.add_argument("--games", type=int, default=100)
    depth.add_argument("--temperature", type=float, default=None)
    depth.add_argument("--min-p", type=float, default=None)
    depth.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    depth.add_argument("--tuned", type=Path, default=Path("data/temperature.json"))

    exp = sub.add_parser("export-examples", help="write fine-tune training/validation examples (never test games)")
    exp.add_argument("--raw", type=Path, default=Path("data/raw"))
    exp.add_argument("--username", default="Justin0921")
    exp.add_argument("--out-dir", type=Path, default=Path("data/finetune"))
    exp.add_argument("--bullet-weight", type=float, default=0.5)
    exp.add_argument("--scrambles", type=Path, default=None,
                     help="leave out the time-scramble moves listed in this file")

    ft = sub.add_parser("finetune", help="fine-tune Maia on the exported examples")
    ft.add_argument("--examples", type=Path, default=Path("data/finetune"))
    ft.add_argument("--model", default="maia3-23m")
    ft.add_argument("--out", type=Path, required=True, help="new checkpoint file (must not exist)")
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--lr", type=float, default=1e-5)
    ft.add_argument("--batch-size", type=int, default=64)
    ft.add_argument("--kl-weight", type=float, default=0.0)
    ft.add_argument("--freeze-layers", type=int, default=0)
    ft.add_argument("--patience", type=int, default=1)
    ft.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    ft.add_argument("--max-examples", type=int, default=None, help="cap for smoke runs")

    cmp_ = sub.add_parser("compare-position", help="base vs fine-tuned top moves for one position")
    common(cmp_)
    cmp_.add_argument("--fen", required=True)
    cmp_.add_argument("--elo", type=int, default=None)
    cmp_.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    cmp_.add_argument("--n", type=int, default=5)

    play = sub.add_parser("play", help="play the bot in the terminal")
    common(play, table=True)
    play.add_argument("--color", choices=["white", "black", "random"], default="white",
                      help="your color")
    play.add_argument("--temperature", type=float, default=None)
    play.add_argument("--min-p", type=float, default=None)
    play.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    play.add_argument("--tuned", type=Path, default=Path("data/temperature.json"))
    play.add_argument("--seed", type=int, default=None)

    blind = sub.add_parser("blind", help="play a hidden model (base, ft1 or ft2), judge if it plays like you, then see the tally")
    blind.add_argument("--raw", type=Path, default=Path("data/raw"))
    blind.add_argument("--table", type=Path, default=Path("data/table.json"))
    blind.add_argument("--model", default="maia3-23m")
    blind.add_argument("--color", choices=["white", "black", "random"], default="white",
                       help="your color")
    blind.add_argument("--min-p", type=float, default=0.05, help="rare-move cutoff, the same for every model")
    blind.add_argument("--seed", type=int, default=None)
    blind.add_argument("--log", type=Path, default=Path("data/blind-log.jsonl"))

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
        elo = args.elo or _json_or(args.calibration, "elo", 1500)
        table = MoveTable.load(args.table) if args.policy == "harness" else None
        policy = HarnessPolicy(table, MaiaPredictor(args.model, checkpoint=args.checkpoint), elo, args.temperature)
        print(f"policy={args.policy} model={args.model} elo={elo} temperature={args.temperature}"
              + (f" scrambles={args.scrambles}" if args.scrambles else ""))
        skip = load_scrambles(args.scrambles) if args.scrambles else None
        _print_summary(summarize(evaluate(policy.distribution, test, args.max_games, skip)))

    elif args.command == "calibrate":
        from chessbot.calibrate import calibrate_elo
        from chessbot.maia import MaiaPredictor

        train, _ = _split(args.raw, args.username)
        start, stop, step = args.elo_range
        best, scores = calibrate_elo(
            MaiaPredictor(args.model, checkpoint=args.checkpoint), train, range(start, stop + 1, step), args.sample_games
        )
        for elo, score in scores.items():
            print(f"{elo:>6}  mean_logprob {score:.4f}{'  <-- best' if elo == best else ''}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"elo": best, "scores": scores}))

    elif args.command == "mark-scrambles":
        games = load_games(args.raw, args.username)
        found: dict[str, list[int]] = {}
        candidates = 0
        with _stockfish() as engine:
            for i, game in enumerate(games, 1):
                candidates += len(scramble_candidates(game, args.max_seconds))
                plies = find_scrambles(engine, game, args.max_seconds, args.loss, args.depth)
                if plies:
                    found[game.url] = plies
                if i % 500 == 0:
                    print(f"{i}/{len(games)} games, {sum(map(len, found.values()))} scramble moves so far", flush=True)
        save_scrambles(found, args.out)
        marked = sum(map(len, found.values()))
        print(f"{marked} scramble moves in {len(found)} games "
              f"(of {candidates} moves with {args.max_seconds:g}s or less left); wrote {args.out}")

    elif args.command == "justin-profile":
        sample = profile_sample(args.raw, args.username, args.games, args.split)
        skip = load_scrambles(args.scrambles) if args.scrambles else None
        with _stockfish() as engine:
            profile = _profile_games(engine, sample, args.depth, skip)
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
        min_p = args.min_p if args.min_p is not None else _json_or(args.tuned, "min_p", 0.0)
        policy = HarnessPolicy(table, MaiaPredictor(args.model, checkpoint=args.checkpoint), elo, temperature, min_p=min_p)
        rng = random.Random(0)
        bot = []
        for _ in range(args.games):
            moves = play_game(policy, rng)
            bot.extend(book_exit_ply(table, moves, c) for c in (chess.WHITE, chess.BLACK))
        print(f"book-exit ply (Justin's turns), Justin test games: n={len(justin)} "
              f"median={statistics.median(justin)} mean={statistics.mean(justin):.1f}")
        print(f"book-exit ply, bot self-play (elo={elo}, T={temperature}): n={len(bot)} "
              f"median={statistics.median(bot)} mean={statistics.mean(bot):.1f}")

    elif args.command == "export-examples":
        from maia3.utils import get_all_possible_moves

        move_index = {m: i for i, m in enumerate(get_all_possible_moves())}
        skip = load_scrambles(args.scrambles) if args.scrambles else None
        n_train, n_val, n_test = export_examples(
            args.raw, args.username, args.out_dir, move_index, args.bullet_weight, skip
        )
        print(f"games: train {n_train}, validation {n_val}, test {n_test} (test not exported)")
        print(f"wrote {args.out_dir}/train.npz and val.npz")

    elif args.command == "finetune":
        from chessbot.examples import load_examples
        from chessbot.finetune import save_checkpoint, train
        from chessbot.maia import MaiaPredictor

        if args.out.exists():
            raise SystemExit(f"{args.out} already exists; choose a new file name")
        train_examples = load_examples(args.examples / "train.npz")
        val_examples = load_examples(args.examples / "val.npz")
        if args.max_examples:
            train_examples = train_examples[:args.max_examples]
            val_examples = val_examples[:max(1, args.max_examples // 5)]
        predictor = MaiaPredictor(args.model, device=args.device)
        print(f"{len(train_examples)} training and {len(val_examples)} validation examples")
        result = train(
            predictor.model, train_examples, val_examples, vocab_size=len(predictor.all_moves),
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, kl_weight=args.kl_weight,
            freeze_layers=args.freeze_layers, patience=args.patience, device=args.device,
        )
        save_checkpoint(result.best_state, args.out)
        print(f"best epoch {result.best_epoch} (0 = never beat the starting model); wrote {args.out}")

    elif args.command == "compare-position":
        from chessbot.compare import format_comparison, top_moves
        from chessbot.maia import MaiaPredictor

        if args.checkpoint is None:
            raise SystemExit("compare-position needs --checkpoint (the fine-tuned weights)")
        elo = args.elo or _json_or(args.calibration, "elo", 1700)
        board = chess.Board(args.fen)
        base = MaiaPredictor(args.model)
        tuned = MaiaPredictor(args.model, checkpoint=args.checkpoint)
        print(format_comparison(board, {
            "base": top_moves(base, board, elo, args.n),
            "fine-tuned": top_moves(tuned, board, elo, args.n),
        }))

    elif args.command == "play":
        from chessbot.maia import MaiaPredictor
        from chessbot.play import choose_color, play_game_vs_human
        from chessbot.policy import HarnessPolicy

        rng = random.Random(args.seed)
        human = choose_color(args.color, rng)
        elo = _json_or(args.calibration, "elo", 1500)
        temperature = args.temperature or _json_or(args.tuned, "temperature", 1.0)
        min_p = args.min_p if args.min_p is not None else _json_or(args.tuned, "min_p", 0.0)
        policy = HarnessPolicy(MoveTable.load(args.table), MaiaPredictor(args.model, checkpoint=args.checkpoint), elo, temperature, min_p=min_p)
        print(f"You are {'White' if human == chess.WHITE else 'Black'}. "
              f"Moves like e4, Nf3, O-O or e2e4. (elo={elo}, T={temperature}, min_p={min_p})")
        result = play_game_vs_human(policy, human, rng)
        print(f"Result: {result}")

    elif args.command == "blind":
        from chessbot.maia import MaiaPredictor
        from chessbot.play import EarlyVerdict, choose_color, load_blind_rows, log_blind, pick_blind, play_game_vs_human, tally_judgments
        from chessbot.policy import HarnessPolicy

        # name -> (checkpoint, calibration, tuned temperature)
        models = {
            "base": (None, "data/calibration.json", "data/temperature.json"),
            "ft1": ("data/finetuned-23m.pt", "data/calibration-ft.json", "data/temperature-ft-minp.json"),
            "ft2": ("data/finetuned-23m-clean.pt", "data/calibration-ft-clean.json", "data/temperature-ft-clean.json"),
        }
        rng = random.Random(args.seed)
        actual = pick_blind(list(models), rng)
        checkpoint, calibration, tuned = models[actual]
        human = choose_color(args.color, rng)
        elo = _json_or(Path(calibration), "elo", 1500)
        temperature = _json_or(Path(tuned), "temperature", 1.0)
        maia = MaiaPredictor(args.model, checkpoint=Path(checkpoint) if checkpoint else None)
        policy = HarnessPolicy(MoveTable.load(args.table), maia, elo, temperature, min_p=args.min_p)
        print(f"Blind test: you are {'White' if human == chess.WHITE else 'Black'}. "
              "The opponent is base, ft1 or ft2 and stays hidden until you judge it. "
              "Moves like e4, Nf3, O-O or e2e4. Press y or n at any time to judge: "
              "does it play like you?")
        try:
            result = play_game_vs_human(policy, human, rng, judging=True)
            print(f"Result: {result}")
            answer = ""
            while answer not in {"y", "n"}:
                answer = input("Did that play like you? (y / n): ").strip().lower()
        except EarlyVerdict as verdict:
            result, answer = "judged early", str(verdict)
        log_blind(args.log, {"actual": actual, "like_me": answer == "y", "result": result,
                             "human": "white" if human == chess.WHITE else "black"})
        print(f"That was {actual}. Tally of games judged \"plays like me\":")
        for name, (yes, total) in tally_judgments(load_blind_rows(args.log), list(models)).items():
            print(f"  {name}: {yes}/{total}")

    elif args.command == "tune-temperature":
        from chessbot.maia import MaiaPredictor
        from chessbot.policy import HarnessPolicy
        from chessbot.selfplay import pick_temperature, play_game

        target = json.loads(args.profile.read_text())["all"]
        elo = _json_or(args.calibration, "elo", 1500)
        maia, table = MaiaPredictor(args.model, checkpoint=args.checkpoint), MoveTable.load(args.table)
        profiles = {}
        with _stockfish() as engine:
            for temperature in args.temperatures:
                policy = HarnessPolicy(table, maia, elo, temperature, min_p=args.min_p)
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
        args.out.write_text(json.dumps({"temperature": best, "min_p": args.min_p,
                                        "profiles": {str(k): v for k, v in profiles.items()}}))
