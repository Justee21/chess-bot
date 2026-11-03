"""Date-based train/test split: the newest games are held out."""
from __future__ import annotations

from typing import Sequence

from chessbot.games import Game


def split_by_date(
    games: Sequence[Game], test_fraction: float = 0.15
) -> tuple[list[Game], list[Game]]:
    if len(games) < 2:
        raise ValueError("need at least two games to split")
    ordered = sorted(games, key=lambda g: g.end_time)
    n_test = max(1, round(len(ordered) * test_fraction))
    return ordered[:-n_test], ordered[-n_test:]
