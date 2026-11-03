"""Position-keyed table of Justin's own moves, built from training games only."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import chess

from chessbot.games import Game


def position_key(board: chess.Board) -> str:
    """Placement, side to move, castling rights, legal en-passant square; no move clocks."""
    return board.epd()


class MoveTable:
    def __init__(self, positions: dict[str, Counter[str]] | None = None):
        self._positions: dict[str, Counter[str]] = positions or {}

    @classmethod
    def build(cls, games: Iterable[Game]) -> "MoveTable":
        positions: dict[str, Counter[str]] = {}
        for game in games:
            board = chess.Board()
            for uci in game.moves:
                if board.turn == game.my_color:
                    positions.setdefault(position_key(board), Counter())[uci] += 1
                board.push_uci(uci)
        return cls(positions)

    def counts(self, board: chess.Board) -> dict[str, int]:
        return dict(self._positions.get(position_key(board), {}))

    def lookup(self, board: chess.Board, min_count: int = 3) -> dict[str, float] | None:
        counts = self.counts(board)
        total = sum(counts.values())
        if total < min_count:
            return None
        return {move: count / total for move, count in counts.items()}

    def __len__(self) -> int:
        return len(self._positions)

    def positions_with_min_count(self, min_count: int = 3) -> int:
        return sum(1 for c in self._positions.values() if sum(c.values()) >= min_count)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: dict(v) for k, v in self._positions.items()}))

    @classmethod
    def load(cls, path: Path) -> "MoveTable":
        raw = json.loads(path.read_text())
        return cls({k: Counter(v) for k, v in raw.items()})
