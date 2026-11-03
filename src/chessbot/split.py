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


def three_way_split(
    games: Sequence[Game], test_fraction: float = 0.15, val_fraction: float = 0.10
) -> tuple[list[Game], list[Game], list[Game]]:
    """(train, val, test) by date: test is the newest slice, val the newest of the rest."""
    if len(games) < 3:
        raise ValueError("need at least three games to split")
    rest, test = split_by_date(games, test_fraction)
    n_val = max(1, round(len(rest) * val_fraction))
    return rest[:-n_val], rest[-n_val:], test
