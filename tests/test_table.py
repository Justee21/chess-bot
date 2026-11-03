import chess

from chessbot.games import Game
from chessbot.table import MoveTable, position_key


def _game(moves, color=chess.WHITE, i=0) -> Game:
    return Game(
        url=f"u{i}", end_time=1000 + i, time_class="blitz",
        my_color=color, my_elo=1500, opp_elo=1500, moves=tuple(moves),
    )


def _after(*ucis: str) -> chess.Board:
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


def test_three_games_same_first_move_gives_certainty():
    table = MoveTable.build([_game(["e2e4", "e7e5"], i=i) for i in range(3)])
    assert table.lookup(chess.Board()) == {"e2e4": 1.0}


def test_two_games_below_min_count_returns_none():
    table = MoveTable.build([_game(["e2e4", "e7e5"], i=i) for i in range(2)])
    assert table.lookup(chess.Board()) is None
    assert table.counts(chess.Board()) == {"e2e4": 2}


def test_opponent_moves_not_counted():
    table = MoveTable.build([_game(["e2e4", "e7e5"], chess.WHITE)])
    assert table.counts(_after("e2e4")) == {}


def test_black_moves_counted_when_justin_is_black():
    table = MoveTable.build([_game(["e2e4", "e7e5"], chess.BLACK)])
    assert table.counts(_after("e2e4")) == {"e7e5": 1}
    assert table.counts(chess.Board()) == {}


def test_lookup_splits_probability_by_frequency():
    games = [_game(["e2e4"], i=0), _game(["e2e4"], i=1), _game(["d2d4"], i=2)]
    dist = MoveTable.build(games).lookup(chess.Board())
    assert dist == {"e2e4": 2 / 3, "d2d4": 1 / 3}


def test_transposition_shares_key():
    a = _after("g1f3", "g8f6", "b1c3")
    b = _after("b1c3", "g8f6", "g1f3")
    assert position_key(a) == position_key(b)


def test_halfmove_clock_ignored():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 10")
    assert position_key(a) == position_key(b)


def test_castling_rights_change_key():
    a = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    assert position_key(a) != position_key(b)


def test_save_load_roundtrip(tmp_path):
    table = MoveTable.build([_game(["e2e4", "e7e5"], i=i) for i in range(3)])
    path = tmp_path / "table.json"
    table.save(path)
    loaded = MoveTable.load(path)
    assert loaded.lookup(chess.Board()) == {"e2e4": 1.0}
