import math

import chess
import pytest

from chessbot.evaluate import (
    MoveResult, classify_phase, evaluate, justin_positions, summarize,
)
from chessbot.games import Game


def _game(moves, color=chess.WHITE, time_class="blitz", i=0) -> Game:
    return Game(url=f"u{i}", end_time=i, time_class=time_class, my_color=color,
                my_elo=1500, opp_elo=1500, moves=tuple(moves))


def test_justin_positions_yields_only_justins_turns():
    white = _game(["e2e4", "e7e5", "g1f3", "b8c6"], chess.WHITE)
    black = _game(["e2e4", "e7e5", "g1f3", "b8c6"], chess.BLACK, i=1)
    from_white = [(b.turn, m) for _, b, m in justin_positions([white])]
    from_black = [(b.turn, m) for _, b, m in justin_positions([black])]
    assert from_white == [(chess.WHITE, "e2e4"), (chess.WHITE, "g1f3")]
    assert from_black == [(chess.BLACK, "e7e5"), (chess.BLACK, "b8c6")]


def test_evaluate_top1_top3_and_logprob():
    dist = {"e2e4": 0.6, "d2d4": 0.4}
    games = [_game(["e2e4"], i=0), _game(["d2d4"], i=1)]
    first, second = evaluate(lambda b: (dist, "maia"), games)
    assert (first.top1, first.top3) == (True, True)
    assert first.logprob == pytest.approx(math.log(0.6))
    assert (second.top1, second.top3) == (False, True)
    assert second.logprob == pytest.approx(math.log(0.4))


def test_zero_probability_move_is_floored():
    (result,) = evaluate(lambda b: ({"d2d4": 1.0}, "maia"), [_game(["e2e4"])])
    assert (result.top1, result.top3) == (False, False)
    assert result.logprob == pytest.approx(math.log(1e-6))


def test_evaluate_max_games_limits_sample():
    games = [_game(["e2e4"], i=i) for i in range(10)]
    results = evaluate(lambda b: ({"e2e4": 1.0}, "maia"), games, max_games=4)
    assert len(results) == 4


def test_summarize_groups_by_phase_time_and_source():
    r = [
        MoveResult(True, True, math.log(0.5), "opening", "bullet", "table"),
        MoveResult(False, True, math.log(0.1), "middlegame", "bullet", "maia"),
        MoveResult(False, False, math.log(0.01), "endgame", "blitz", "maia"),
    ]
    s = summarize(r)
    assert s["all"]["n"] == 3
    assert s["all"]["top1"] == pytest.approx(1 / 3)
    assert s["all"]["top3"] == pytest.approx(2 / 3)
    assert s["phase:opening"]["top1"] == 1.0
    assert s["time:bullet"]["n"] == 2
    assert s["source:maia"]["n"] == 2
    assert s["source:table"]["mean_logprob"] == pytest.approx(math.log(0.5))


def test_classify_phase_cases():
    assert classify_phase(chess.Board()) == "opening"
    middle = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 11")
    assert classify_phase(middle) == "middlegame"
    assert classify_phase(chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 30")) == "endgame"
    assert classify_phase(chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 5")) == "opening"
