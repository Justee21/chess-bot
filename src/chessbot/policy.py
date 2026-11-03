"""Table-first, Maia-fallback move policy. Moves are sampled, never argmax; the bot never resigns."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import chess

from chessbot.table import MoveTable


class MovePredictor(Protocol):
    def move_probs(self, board: chess.Board, self_elo: int, oppo_elo: int) -> dict[str, float]: ...


@dataclass(frozen=True)
class Choice:
    move: chess.Move
    source: str  # "table" | "maia"


class HarnessPolicy:
    def __init__(
        self,
        table: MoveTable | None,
        maia: MovePredictor,
        elo: int,
        temperature: float,
        min_count: int = 3,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.table = table
        self.maia = maia
        self.elo = elo
        self.temperature = temperature
        self.min_count = min_count

    def distribution(self, board: chess.Board) -> tuple[dict[str, float], str]:
        if not any(board.legal_moves):
            raise ValueError("no legal moves in this position")
        if self.table is not None:
            from_table = self.table.lookup(board, self.min_count)
            if from_table:
                return from_table, "table"
        probs = self.maia.move_probs(board, self.elo, self.elo)
        if self.temperature != 1.0:
            probs = {m: p ** (1.0 / self.temperature) for m, p in probs.items()}
        total = sum(probs.values())
        return {m: p / total for m, p in probs.items()}, "maia"

    def choose(self, board: chess.Board, rng: random.Random) -> Choice:
        dist, source = self.distribution(board)
        uci = rng.choices(list(dist), weights=list(dist.values()))[0]
        return Choice(chess.Move.from_uci(uci), source)
