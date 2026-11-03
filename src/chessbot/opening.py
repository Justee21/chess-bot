"""How deep Justin (or the bot) stays inside the move table."""
from __future__ import annotations

from typing import Sequence

import chess

from chessbot.table import MoveTable


def book_exit_ply(
    table: MoveTable, moves: Sequence[str], my_color: chess.Color, min_count: int = 3
) -> int:
    """Ply of the first of Justin's turns not covered by the table; len(moves) if never."""
    board = chess.Board()
    for ply, uci in enumerate(moves):
        if board.turn == my_color and table.lookup(board, min_count) is None:
            return ply
        board.push_uci(uci)
    return len(moves)
