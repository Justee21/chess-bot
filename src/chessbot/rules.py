"""When a game has actually ended (chess.com draws automatically; no one-move-ahead draw claims)."""
from __future__ import annotations

import chess


def is_finished(board: chess.Board) -> bool:
    """Mate, stalemate, insufficient material, fivefold/75-move, a real threefold, or 50 moves."""
    return board.is_game_over() or board.is_repetition(3) or board.halfmove_clock >= 100
