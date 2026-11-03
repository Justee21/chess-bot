import chess

from chessbot.games import Game
from chessbot.opening import book_exit_ply
from chessbot.table import MoveTable


def _game(moves, color=chess.WHITE, i=0) -> Game:
    return Game(url=f"u{i}", end_time=i, time_class="blitz", my_color=color,
                my_elo=1500, opp_elo=1500, moves=tuple(moves))


def _table_covering_only_the_start() -> MoveTable:
    return MoveTable.build([_game(["e2e4", "e7e5"], i=i) for i in range(3)])


def test_exit_ply_when_table_runs_out():
    table = _table_covering_only_the_start()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert book_exit_ply(table, moves, chess.WHITE) == 2


def test_never_leaves_book_returns_game_length():
    table = _table_covering_only_the_start()
    assert book_exit_ply(table, ["e2e4", "e7e5"], chess.WHITE) == 2


def test_only_justins_turns_count():
    empty = MoveTable.build([])
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    # ply 0 is the opponent's (uncovered) move; the first of Justin's turns is ply 1
    assert book_exit_ply(empty, moves, chess.BLACK) == 1
