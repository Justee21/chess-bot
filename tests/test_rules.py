import chess

from chessbot.rules import is_finished

SEVEN = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]


def _after(ucis) -> chess.Board:
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


def test_ordinary_position_is_not_finished():
    assert not is_finished(chess.Board())


def test_checkmate_is_finished():
    assert is_finished(_after(["f2f3", "e7e5", "g2g4", "d8h4"]))


def test_stalemate_and_insufficient_material_are_finished():
    assert is_finished(chess.Board("k7/8/KQ6/8/8/8/8/8 b - - 0 1"))
    assert is_finished(chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1"))


def test_claimable_but_not_yet_repeated_is_not_finished():
    assert not is_finished(_after(SEVEN))


def test_real_threefold_repetition_is_finished():
    assert is_finished(_after(SEVEN + ["f6g8"]))


def test_fifty_move_rule_is_finished():
    assert is_finished(chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 100 80"))
