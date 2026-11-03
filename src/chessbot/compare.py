"""Side-by-side view of what different Maia models would play in one position."""
from __future__ import annotations

import chess

from chessbot.policy import MovePredictor


def top_moves(predictor: MovePredictor, board: chess.Board, elo: int, n: int = 5) -> list[tuple[str, float]]:
    """The model's `n` likeliest moves as (SAN, probability), most likely first."""
    probs = predictor.move_probs(board, elo, elo)
    ranked = sorted(probs, key=lambda m: -probs[m])[:n]
    return [(board.san(chess.Move.from_uci(m)), probs[m]) for m in ranked]


def format_comparison(board: chess.Board, columns: dict[str, list[tuple[str, float]]]) -> str:
    """One block: the position, then a column of ranked moves per model."""
    names = list(columns)
    rows = max((len(v) for v in columns.values()), default=0)
    lines = [board.fen(), "", "    " + "".join(f"{name:<18}" for name in names)]
    for i in range(rows):
        cells = []
        for name in names:
            moves = columns[name]
            cells.append(f"{moves[i][0]:<7}{moves[i][1]:>6.1%}     " if i < len(moves) else " " * 18)
        lines.append(f"{i + 1:>2}  " + "".join(cells))
    return "\n".join(lines)
