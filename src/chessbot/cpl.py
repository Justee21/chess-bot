"""Centipawn loss and error profile, measured with a UCI engine such as Stockfish."""
from __future__ import annotations

from typing import Collection, Sequence

import chess
import chess.engine

from chessbot.rules import is_finished

MATE_SCORE = 1000
LOSS_CAP = 1000.0


def _white_eval(engine, board: chess.Board, depth: int) -> int:
    """Centipawns from White's view; finished games are scored without the engine."""
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if is_finished(board):
        return 0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=MATE_SCORE)


def move_loss(engine, board: chess.Board, uci: str, depth: int = 12) -> float:
    """Centipawn loss (capped) of playing `uci` on `board`, from the mover's view."""
    before = _white_eval(engine, board, depth)
    after_board = board.copy()
    after_board.push_uci(uci)
    after = _white_eval(engine, after_board, depth)
    drop = before - after if board.turn == chess.WHITE else after - before
    return min(max(0.0, float(drop)), LOSS_CAP)


def game_cpls(
    engine, moves: Sequence[str], color: chess.Color, depth: int = 12, skip: Collection[int] = ()
) -> list[float]:
    """Centipawn loss of each move played by `color`, one engine call per position.
    `skip` holds ply indices (0-based) to leave out."""
    board = chess.Board()
    evals = [_white_eval(engine, board, depth)]
    for uci in moves:
        board.push_uci(uci)
        evals.append(_white_eval(engine, board, depth))
    losses = []
    for i in range(len(moves)):
        mover = chess.WHITE if i % 2 == 0 else chess.BLACK
        if mover != color or i in skip:
            continue
        drop = evals[i] - evals[i + 1] if mover == chess.WHITE else evals[i + 1] - evals[i]
        losses.append(min(max(0.0, float(drop)), LOSS_CAP))
    return losses


def error_profile(cpls: Sequence[float]) -> dict[str, float]:
    n = len(cpls)
    if n == 0:
        return {"n": 0, "avg_cpl": 0.0, "p100": 0.0, "p200": 0.0, "p300": 0.0}
    return {
        "n": n,
        "avg_cpl": sum(cpls) / n,
        "p100": sum(c > 100 for c in cpls) / n,
        "p200": sum(c > 200 for c in cpls) / n,
        "p300": sum(c > 300 for c in cpls) / n,
    }
