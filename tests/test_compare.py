import chess

from chessbot.compare import format_comparison, top_moves


class Stub:
    def __init__(self, probs):
        self.probs = probs

    def move_probs(self, board, self_elo, oppo_elo):
        return dict(self.probs)


def test_top_moves_sorted_and_in_san():
    board = chess.Board()
    stub = Stub({"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.15, "a2a3": 0.05})
    assert top_moves(stub, board, 1500, n=3) == [("e4", 0.5), ("d4", 0.3), ("Nf3", 0.15)]


def test_format_comparison_has_a_column_per_model():
    board = chess.Board()
    text = format_comparison(board, {
        "base": [("e4", 0.5), ("d4", 0.3)],
        "fine-tuned": [("d4", 0.6)],
    })
    assert "base" in text and "fine-tuned" in text
    assert "e4" in text and "50.0%" in text and "60.0%" in text
    assert len(text.splitlines()) >= 3
