"""Table-first, Maia-fallback move policy. Moves are sampled, never argmax; the bot never resigns or draws."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import chess

from chessbot.rules import move_draws
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
        min_p: float = 0.0,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError("min_p must be between 0 and 1")
        self.table = table
        self.maia = maia
        self.elo = elo
        self.temperature = temperature
        self.min_count = min_count
        self.min_p = min_p

    def distribution(self, board: chess.Board) -> tuple[dict[str, float], str]:
        if not any(board.legal_moves):
            raise ValueError("no legal moves in this position")
        if self.table is not None:
            from_table = self.table.lookup(board, self.min_count)
            if from_table:
                return from_table, "table"
        return self._maia_distribution(board), "maia"

    def _maia_distribution(self, board: chess.Board) -> dict[str, float]:
        probs = self.maia.move_probs(board, self.elo, self.elo)
        if self.temperature != 1.0:
            probs = {m: p ** (1.0 / self.temperature) for m, p in probs.items()}
        total = sum(probs.values())
        return {m: p / total for m, p in probs.items()}

    def _playable(self, board: chess.Board, dist: dict[str, float]) -> dict[str, float]:
        """`dist` without moves that end the game in a draw; empty if nothing is left."""
        return {
            m: p for m, p in dist.items()
            if p > 0 and not move_draws(board, chess.Move.from_uci(m))
        }

    def _trim(self, pool: dict[str, float]) -> dict[str, float]:
        """Drop moves whose probability is under `min_p` times the best move's, so a rare
        slip is not sampled when one move is clearly right (the best move always stays)."""
        cutoff = self.min_p * max(pool.values())
        return {m: p for m, p in pool.items() if p >= cutoff}

    def choose(self, board: chess.Board, rng: random.Random) -> Choice:
        """Sample a move. The bot never plays a move that draws the game unless every move does,
        and never one far less likely than the best (`min_p`)."""
        dist, source = self.distribution(board)
        pool = self._playable(board, dist)
        if not pool and source == "table":
            dist, source = self._maia_distribution(board), "maia"
            pool = self._playable(board, dist)
        if not pool:  # Maia gave no non-drawing move weight; any non-drawing legal move beats a draw
            legal = {m.uci(): 1.0 for m in board.legal_moves}
            pool = self._playable(board, legal) or dist
        pool = self._trim(pool)
        uci = rng.choices(list(pool), weights=list(pool.values()))[0]
        return Choice(chess.Move.from_uci(uci), source)
