import chess
import chess.engine

from chessbot.games import Game
from chessbot.scramble import find_scrambles, load_scrambles, save_scrambles, scramble_candidates


class StubEngine:
    """White-pov centipawn score per position, indexed by plies played."""

    def __init__(self, white_cps):
        self.cps = white_cps

    def analyse(self, board, limit):
        cp = self.cps[len(board.move_stack)]
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}

MOVES = ("e2e4", "e7e5", "g1f3", "b8c6")


def _game(clocks, time_class="bullet", color=chess.WHITE, url="u") -> Game:
    return Game(url=url, end_time=1, time_class=time_class, my_color=color, my_elo=1500,
                opp_elo=1500, moves=MOVES, clocks=tuple(clocks))


def test_candidates_are_my_moves_with_five_seconds_or_less_left():
    assert scramble_candidates(_game([50, 49, 5.0, 4.0])) == [2]


def test_black_candidates_use_black_plies():
    assert scramble_candidates(_game([50, 49, 30, 4.9], color=chess.BLACK)) == [3]


def test_only_bullet_games_have_candidates():
    assert scramble_candidates(_game([50, 49, 2, 1], time_class="blitz")) == []


def test_game_without_clocks_has_no_candidates():
    assert scramble_candidates(_game([])) == []


def test_find_scrambles_flags_only_big_losses():
    game = _game([50, 49, 3, 2])
    assert find_scrambles(StubEngine([20, 20, 20, -250, -250]), game) == [2]  # lost 270
    assert find_scrambles(StubEngine([20, 20, 20, -80, -80]), game) == []  # lost 100


def test_find_scrambles_ignores_moves_with_plenty_of_time():
    game = _game([50, 49, 40, 2])  # my only candidate would be ply 2, but it had 40s
    assert find_scrambles(StubEngine([20, 20, 20, -250, -250]), game) == []


def test_scrambles_roundtrip(tmp_path):
    path = tmp_path / "scrambles.json"
    save_scrambles({"a": [2, 5], "b": []}, path)
    assert load_scrambles(path) == {"a": {2, 5}, "b": set()}
