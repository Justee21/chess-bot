import random

import chess
import pytest

from chessbot.games import Game
from chessbot.split import split_by_date, three_way_split


def _game(i: int) -> Game:
    return Game(
        url=f"u{i}", end_time=1000 + i, time_class="blitz",
        my_color=chess.WHITE, my_elo=1500, opp_elo=1500, moves=("e2e4",),
    )


def _shuffled(n: int) -> list[Game]:
    games = [_game(i) for i in range(n)]
    random.Random(0).shuffle(games)
    return games


def test_test_set_is_newest():
    train, test = split_by_date(_shuffled(20))
    assert len(test) == 3
    assert max(g.end_time for g in train) <= min(g.end_time for g in test)


def test_partition_is_complete_and_disjoint():
    games = _shuffled(20)
    train, test = split_by_date(games)
    assert {g.url for g in train} | {g.url for g in test} == {g.url for g in games}
    assert not {g.url for g in train} & {g.url for g in test}


def test_fewer_than_two_games_raises_valueerror():
    with pytest.raises(ValueError):
        split_by_date([_game(0)])


def test_three_way_split_is_ordered_and_disjoint():
    games = _shuffled(40)
    train, val, test = three_way_split(games)
    assert (len(train), len(val), len(test)) == (31, 3, 6)
    assert max(g.end_time for g in train) < min(g.end_time for g in val)
    assert max(g.end_time for g in val) < min(g.end_time for g in test)
    assert {g.url for g in train + val + test} == {g.url for g in games}
    assert test == split_by_date(games)[1]


def test_three_way_split_needs_three_games():
    with pytest.raises(ValueError):
        three_way_split(_shuffled(2))
