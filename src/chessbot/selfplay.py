"""Bot-vs-bot games and temperature selection. The bot never resigns."""
from __future__ import annotations

import random

import chess

from chessbot.policy import HarnessPolicy
from chessbot.rules import is_finished


def play_game(
    policy: HarnessPolicy, rng: random.Random, max_plies: int = 400
) -> tuple[str, ...]:
    """Play `policy` against itself until the game is actually finished or `max_plies`."""
    board = chess.Board()
    moves: list[str] = []
    while len(moves) < max_plies and not is_finished(board):
        move = policy.choose(board, rng).move
        board.push(move)
        moves.append(move.uci())
    return tuple(moves)


def pick_temperature(
    profiles: dict[float, dict[str, float]], target: dict[str, float]
) -> float:
    """The temperature whose self-play avg_cpl is closest to Justin's."""
    return min(profiles, key=lambda t: abs(profiles[t]["avg_cpl"] - target["avg_cpl"]))
