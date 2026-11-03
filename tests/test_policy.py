import random

import chess
import pytest

from chessbot.games import Game
from chessbot.policy import Choice, HarnessPolicy
from chessbot.table import MoveTable


class StubMaia:
    def __init__(self, probs):
        self.probs = probs
        self.calls = []

    def move_probs(self, board, self_elo, oppo_elo):
        self.calls.append((self_elo, oppo_elo))
        return dict(self.probs)


def _table(first_move: str, n: int) -> MoveTable:
    games = [
        Game(url=f"u{i}", end_time=i, time_class="blitz", my_color=chess.WHITE,
             my_elo=1500, opp_elo=1500, moves=(first_move,))
        for i in range(n)
    ]
    return MoveTable.build(games)


MAIA = {"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.2}


def test_table_hit_uses_table():
    policy = HarnessPolicy(_table("c2c4", 3), StubMaia(MAIA), elo=1500, temperature=1.0)
    dist, source = policy.distribution(chess.Board())
    assert (dist, source) == ({"c2c4": 1.0}, "table")


def test_table_miss_falls_back_to_maia():
    policy = HarnessPolicy(_table("c2c4", 2), StubMaia(MAIA), elo=1500, temperature=1.0)
    dist, source = policy.distribution(chess.Board())
    assert source == "maia"
    assert dist == pytest.approx(MAIA)


def test_no_table_uses_maia():
    maia = StubMaia(MAIA)
    policy = HarnessPolicy(None, maia, elo=1400, temperature=1.0)
    _, source = policy.distribution(chess.Board())
    assert source == "maia"
    assert maia.calls == [(1400, 1400)]


def test_temperature_one_leaves_distribution_unchanged():
    policy = HarnessPolicy(None, StubMaia(MAIA), elo=1500, temperature=1.0)
    dist, _ = policy.distribution(chess.Board())
    assert dist == pytest.approx(MAIA)


def test_lower_temperature_sharpens_and_still_sums_to_one():
    policy = HarnessPolicy(None, StubMaia(MAIA), elo=1500, temperature=0.5)
    dist, _ = policy.distribution(chess.Board())
    assert dist["e2e4"] > 0.5
    assert sum(dist.values()) == pytest.approx(1.0)


def test_choose_samples_not_argmax():
    maia = StubMaia({"e2e4": 0.5, "d2d4": 0.5})
    policy = HarnessPolicy(None, maia, elo=1500, temperature=1.0)
    rng = random.Random(0)
    picks = {policy.choose(chess.Board(), rng).move.uci() for _ in range(500)}
    assert picks == {"e2e4", "d2d4"}


def test_choose_returns_choice_with_move_and_source():
    policy = HarnessPolicy(_table("c2c4", 3), StubMaia(MAIA), elo=1500, temperature=1.0)
    choice = policy.choose(chess.Board(), random.Random(0))
    assert choice == Choice(chess.Move.from_uci("c2c4"), "table")


def test_choose_with_no_legal_moves_raises_valueerror():
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):  # fool's mate
        board.push_uci(uci)
    policy = HarnessPolicy(None, StubMaia({}), elo=1500, temperature=1.0)
    with pytest.raises(ValueError):
        policy.choose(board, random.Random(0))


STALEMATE_TRAP = "k7/8/1K6/8/8/8/8/2Q5 w - - 0 1"  # Qc7 would stalemate Black


def test_choose_never_stalemates_when_another_move_exists():
    board = chess.Board(STALEMATE_TRAP)
    policy = HarnessPolicy(None, StubMaia({"c1c7": 0.9, "c1c2": 0.1}), elo=1500, temperature=1.0)
    rng = random.Random(0)
    picks = {policy.choose(board, rng).move.uci() for _ in range(300)}
    assert picks == {"c1c2"}


def test_choose_never_repeats_a_position_a_third_time():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        board.push_uci(uci)  # f6g8 now would be the third occurrence of the start position
    policy = HarnessPolicy(None, StubMaia({"f6g8": 0.9, "e7e5": 0.1}), elo=1500, temperature=1.0)
    rng = random.Random(0)
    picks = {policy.choose(board, rng).move.uci() for _ in range(300)}
    assert picks == {"e7e5"}


def test_table_move_that_draws_is_skipped_for_maia():
    class DrawingTable:
        def lookup(self, board, min_count):
            return {"c1c7": 1.0}

    board = chess.Board(STALEMATE_TRAP)
    policy = HarnessPolicy(DrawingTable(), StubMaia({"c1c2": 1.0}), elo=1500, temperature=1.0)
    choice = policy.choose(board, random.Random(0))
    assert choice == Choice(chess.Move.from_uci("c1c2"), "maia")


def test_distribution_is_not_filtered_so_evaluation_stays_honest():
    board = chess.Board(STALEMATE_TRAP)
    policy = HarnessPolicy(None, StubMaia({"c1c7": 0.9, "c1c2": 0.1}), elo=1500, temperature=1.0)
    dist, _ = policy.distribution(board)
    assert dist["c1c7"] == pytest.approx(0.9)


def test_nonpositive_temperature_rejected():
    with pytest.raises(ValueError):
        HarnessPolicy(None, StubMaia(MAIA), elo=1500, temperature=0)
