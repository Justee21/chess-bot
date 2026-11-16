"""UCI transport for the frozen policy. No search, strength override, or fallback model."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import random
import sys
from typing import TextIO

import chess

from chessbot.frozen import load_policy
from chessbot.policy import HarnessPolicy


def parse_position(tokens: list[str]) -> chess.Board:
    if not tokens:
        raise ValueError("position needs startpos or fen")
    if tokens[0] == "startpos":
        board, rest = chess.Board(), tokens[1:]
    elif tokens[0] == "fen" and len(tokens) >= 7:
        board, rest = chess.Board(" ".join(tokens[1:7])), tokens[7:]
    else:
        raise ValueError("invalid position command")
    if not board.is_valid():
        raise ValueError("invalid chess position")
    if rest:
        if rest[0] != "moves":
            raise ValueError("expected moves after position")
        for move in rest[1:]:
            board.push_uci(move)
    return board


def run(policy: HarnessPolicy, source: TextIO, sink: TextIO, rng: random.Random) -> None:
    board = chess.Board()

    def send(line: str) -> None:
        print(line, file=sink, flush=True)

    for line in source:
        tokens = line.split()
        if not tokens:
            continue
        command, args = tokens[0], tokens[1:]
        if command == "uci":
            send("id name Justin Chess (frozen fine-tune 2)")
            send("id author Justin Roberts; Maia 3 by UofT CSSLab")
            send("uciok")
        elif command == "isready":
            send("readyok")
        elif command == "ucinewgame":
            board = chess.Board()
        elif command == "position":
            # Fail on malformed history rather than playing from a stale board.
            board = parse_position(args)
        elif command == "go":
            if any(arg in args for arg in ("ponder", "infinite", "searchmoves")):
                raise ValueError("ponder, infinite, and restricted searches are unsupported")
            if not any(board.legal_moves):
                send("bestmove 0000")
            else:
                move = policy.choose(board, rng).move
                if move not in board.legal_moves:
                    raise ValueError("policy returned an illegal move")
                send(f"bestmove {move.uci()}")
        elif command == "quit":
            break
        # stop is a no-op: each go returns one sampled move immediately.
        # No options are advertised, so a GUI cannot override frozen settings.
        elif command in {"stop", "setoption", "debug", "register"}:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/final.json"))
    args = parser.parse_args()
    try:
        # Keep upstream model diagnostics away from the UCI stdout channel.
        with redirect_stdout(sys.stderr):
            import torch
            torch.set_num_threads(2)
            policy = load_policy(args.config)
        run(policy, sys.stdin, sys.stdout, random.Random())
    except (OSError, ValueError, KeyError) as error:
        print(f"chessbot: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
