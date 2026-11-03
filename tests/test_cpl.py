import chess
import chess.engine
import pytest

from chessbot.cpl import error_profile, game_cpls


class StubEngine:
    """White-pov centipawn score per position, indexed by plies played."""

    def __init__(self, white_cps):
        self.cps = white_cps

    def analyse(self, board, limit):
        assert not board.is_game_over(), "engine must not be asked about finished games"
        cp = self.cps[len(board.move_stack)]
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}


def test_error_profile_numbers():
    p = error_profile([0, 50, 150, 250, 400])
    assert p["n"] == 5
    assert p["avg_cpl"] == pytest.approx(170)
    assert p["p100"] == pytest.approx(0.6)
    assert p["p200"] == pytest.approx(0.4)
    assert p["p300"] == pytest.approx(0.2)


def test_error_profile_empty_is_zeros():
    assert error_profile([]) == {"n": 0, "avg_cpl": 0.0, "p100": 0.0, "p200": 0.0, "p300": 0.0}


def test_game_cpls_zero_when_eval_holds():
    engine = StubEngine([30, 30, 30])
    assert game_cpls(engine, ["e2e4", "e7e5"], chess.WHITE) == [0.0]


def test_game_cpls_reports_drop_for_the_right_color():
    engine = StubEngine([20, -230, -230])
    assert game_cpls(engine, ["e2e4", "e7e5"], chess.WHITE) == [250.0]
    assert game_cpls(engine, ["e2e4", "e7e5"], chess.BLACK) == [0.0]


def test_black_loss_is_measured_from_blacks_view():
    engine = StubEngine([0, 0, 180])  # black's move lets white's eval rise by 180
    assert game_cpls(engine, ["e2e4", "e7e5"], chess.BLACK) == [180.0]


def test_cpl_capped_at_1000():
    engine = StubEngine([0, -3000, -3000])
    assert game_cpls(engine, ["e2e4", "e7e5"], chess.WHITE) == [1000.0]


def test_finished_game_scored_without_asking_engine_about_mate():
    engine = StubEngine([0, 0, 0, -1000])  # mate in 1 for Black; index 4 (the mate) is never read
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]  # fool's mate
    assert game_cpls(engine, moves, chess.WHITE) == [0.0, 1000.0]


@pytest.mark.slow
def test_real_stockfish_flags_the_blunder():
    import shutil

    path = shutil.which("stockfish")
    assert path, "install Stockfish (brew install stockfish)"
    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        losses = game_cpls(engine, ["f2f3", "e7e5", "g2g4", "d8h4"], chess.WHITE, depth=8)
    assert losses[1] >= 500
