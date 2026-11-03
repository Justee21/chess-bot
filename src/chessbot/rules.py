"""When a game has actually ended (chess.com draws automatically; no one-move-ahead draw claims)."""
from __future__ import annotations

import chess


def move_draws(board: chess.Board, move: chess.Move) -> bool:
    """True if playing `move` ends the game in a draw (stalemate, dead position, threefold, 50-move)."""
    board.push(move)
    try:
        return is_finished(board) and not board.is_checkmate()
    finally:
        board.pop()


def is_finished(board: chess.Board) -> bool:
    """Mate, stalemate, insufficient material, fivefold/75-move, a real threefold, or 50 moves."""
    return board.is_game_over() or board.is_repetition(3) or board.halfmove_clock >= 100
