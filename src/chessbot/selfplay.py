"""Bot-vs-bot games and temperature selection. The bot never resigns."""
from __future__ import annotations

import random

import chess

from chessbot.policy import HarnessPolicy


def play_game(
    policy: HarnessPolicy, rng: random.Random, max_plies: int = 400
) -> tuple[str, ...]:
    """Play `policy` against itself until a chess outcome (draws claimable) or `max_plies`."""
    board = chess.Board()
    moves: list[str] = []
    while len(moves) < max_plies and board.outcome(claim_draw=True) is None:
        move = policy.choose(board, rng).move
        board.push(move)
        moves.append(move.uci())
    return tuple(moves)


def pick_temperature(
    profiles: dict[float, dict[str, float]], target: dict[str, float]
) -> float:
    """The temperature whose self-play avg_cpl is closest to Justin's."""
    return min(profiles, key=lambda t: abs(profiles[t]["avg_cpl"] - target["avg_cpl"]))
