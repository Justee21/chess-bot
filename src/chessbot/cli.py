from __future__ import annotations

import argparse
from pathlib import Path

from chessbot.games import download_archives


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="chessbot")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download chess.com monthly archives")
    download.add_argument("--out", type=Path, default=Path("data/raw"))
    download.add_argument("--username", default="Justin0921")

    args = parser.parse_args(argv)
    if args.command == "download":
        months = download_archives(args.username, args.out)
        print(f"wrote {months} monthly archives to {args.out}")
