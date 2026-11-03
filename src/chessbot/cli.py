from __future__ import annotations

import argparse
from pathlib import Path

from chessbot.games import download_archives, load_games
from chessbot.split import split_by_date
from chessbot.table import MoveTable


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chessbot")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download chess.com monthly archives")
    download.add_argument("--out", type=Path, default=Path("data/raw"))
    download.add_argument("--username", default="Justin0921")

    build = sub.add_parser("build-table", help="build the move table from training games")
    build.add_argument("--raw", type=Path, default=Path("data/raw"))
    build.add_argument("--out", type=Path, default=Path("data/table.json"))
    build.add_argument("--username", default="Justin0921")

    args = parser.parse_args(argv)
    if args.command == "download":
        months = download_archives(args.username, args.out)
        print(f"wrote {months} monthly archives to {args.out}")
    elif args.command == "build-table":
        train, test = split_by_date(load_games(args.raw, args.username))
        table = MoveTable.build(train)
        table.save(args.out)
        print(f"train games: {len(train)}, test games: {len(test)}")
        print(f"positions in table: {len(table)}, seen >= 3 times: {table.positions_with_min_count(3)}")
        print(f"wrote {args.out}")
