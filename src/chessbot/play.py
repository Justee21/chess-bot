"""Play the harness bot in a terminal. The bot never resigns and never draws."""
from __future__ import annotations

import random
from typing import Callable

import chess

from chessbot.policy import HarnessPolicy
from chessbot.rules import is_finished


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
) -> str:
    """Play until the game ends or the human types quit. Returns "1-0", "0-1", "1/2-1/2" or "*"."""
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
            write(f"Bot plays {san}  ({note})")
        write(board.unicode(borders=True, orientation=human))
    write(_ending(board))
    return board.result(claim_draw=True)
