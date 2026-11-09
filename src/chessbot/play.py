"""Play the harness bot in a terminal. The bot never resigns and never draws."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Sequence

import chess

from chessbot.policy import HarnessPolicy
from chessbot.rules import is_finished


class EarlyVerdict(Exception):
    """The human pressed y or n mid-game to judge the bot before the game ended."""

    def __init__(self, like_me: bool):
        super().__init__("y" if like_me else "n")
        self.like_me = like_me


def choose_color(color: str, rng: random.Random) -> chess.Color:
    """The human's color from "white", "black" or "random" (chess.BLACK is falsy, so compare strings)."""
    if color == "random":
        return rng.choice([chess.WHITE, chess.BLACK])
    return chess.WHITE if color == "white" else chess.BLACK


def pick_blind(names: Sequence[str], rng: random.Random) -> str:
    """The hidden opponent for a blind test."""
    return rng.choice(list(names))


def log_blind(path: Path, entry: dict) -> None:
    """Append one blind-test result (actual model, guess, result) as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_blind_rows(path: Path) -> list[dict]:
    """Every logged blind-test row; empty if nothing has been logged yet."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tally_judgments(rows: Sequence[dict], names: Sequence[str]) -> dict[str, tuple[int, int]]:
    """(games judged "plays like me", games judged) per model; rows without a judgment are skipped."""
    out = {name: [0, 0] for name in names}
    for row in rows:
        if row.get("actual") in out and "like_me" in row:
            out[row["actual"]][1] += 1
            out[row["actual"]][0] += bool(row["like_me"])
    return {name: (yes, total) for name, (yes, total) in out.items()}


def parse_move(board: chess.Board, text: str) -> chess.Move:
    """A legal move from SAN ("Nf3", "e4") or UCI ("g1f3"); ValueError if it is neither."""
    text = text.strip()
    if not text:
        raise ValueError("empty move")
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text.lower())
    except ValueError:
        raise ValueError(f"{text!r} is not a legal move") from None
    if move not in board.legal_moves:
        raise ValueError(f"{text!r} is not a legal move")
    return move


def _ending(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate. {winner} wins."
    return "Draw."


def play_game_vs_human(
    policy: HarnessPolicy,
    human: chess.Color,
    rng: random.Random,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    judging: bool = False,
) -> str:
    """Play until the game ends or the human types quit (when `judging`, y or n raises EarlyVerdict). Returns "1-0", "0-1", "1/2-1/2" or "*"."""
    board = chess.Board()
    write(board.unicode(borders=True, orientation=human))
    while not is_finished(board):
        if board.turn == human:
            try:
                text = read("Your move (quit to stop): ")
            except EOFError:
                return "*"
            if text.strip().lower() in {"quit", "exit"}:
                return "*"
            if judging and text.strip().lower() in {"y", "n"}:
                raise EarlyVerdict(text.strip().lower() == "y")
            try:
                move = parse_move(board, text)
            except ValueError:
                write(f"{text.strip()!r} is not a legal move, try again.")
                continue
            board.push(move)
        else:
            choice = policy.choose(board, rng)
            san = board.san(choice.move)
            board.push(choice.move)
            note = "from your opening book" if choice.source == "table" else "Maia"
            write(f"Bot plays {san}" if judging else f"Bot plays {san}  ({note})")
        write(board.unicode(borders=True, orientation=human))
    write(_ending(board))
    return board.result(claim_draw=True)
