"""Time-scramble moves: bullet moves played with almost no clock left that also lose a lot.
They reflect running out of time, not how Justin plays, so training and metrics can leave them out."""
from __future__ import annotations

import json
from pathlib import Path

import chess

from chessbot.cpl import move_loss
from chessbot.games import Game


def scramble_candidates(game: Game, max_seconds: float = 5.0) -> list[int]:
    """Ply indices of Justin's moves in a bullet game that left him `max_seconds` or less."""
    if game.time_class != "bullet" or len(game.clocks) != len(game.moves):
        return []
    return [
        ply for ply in range(len(game.moves))
        if (ply % 2 == 0) == (game.my_color == chess.WHITE) and game.clocks[ply] <= max_seconds
    ]


def find_scrambles(
    engine, game: Game, max_seconds: float = 5.0, loss_cp: float = 200.0, depth: int = 12
) -> list[int]:
    """The candidates that also lost at least `loss_cp` centipawns."""
    candidates = scramble_candidates(game, max_seconds)
    if not candidates:
        return []
    board = chess.Board()
    found = []
    for ply, uci in enumerate(game.moves):
        if ply in candidates and move_loss(engine, board, uci, depth) >= loss_cp:
            found.append(ply)
        board.push_uci(uci)
    return found


def save_scrambles(scrambles: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrambles))


def load_scrambles(path: Path) -> dict[str, set[int]]:
    return {url: set(plies) for url, plies in json.loads(path.read_text()).items()}
