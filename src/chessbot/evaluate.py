"""Move-match evaluation: how well a policy predicts the moves Justin actually played."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Collection, Iterable, Iterator, Mapping, Sequence

import chess

from chessbot.games import Game

LOGPROB_FLOOR = 1e-6


@dataclass(frozen=True)
class MoveResult:
    top1: bool
    top3: bool
    logprob: float
    phase: str
    time_class: str
    source: str


def classify_phase(board: chess.Board) -> str:
    if board.fullmove_number <= 10:
        return "opening"
    non_king_pieces = chess.popcount(board.occupied) - 2
    return "endgame" if non_king_pieces <= 10 else "middlegame"


def justin_positions(
    games: Iterable[Game], skip: Mapping[str, Collection[int]] | None = None
) -> Iterator[tuple[Game, chess.Board, str]]:
    """Board before each of Justin's moves (with move history), and the UCI he played.
    Plies listed for a game's url in `skip` are left out."""
    for game in games:
        board = chess.Board()
        skipped = (skip or {}).get(game.url, ())
        for ply, uci in enumerate(game.moves):
            if board.turn == game.my_color and ply not in skipped:
                yield game, board.copy(), uci
            board.push_uci(uci)


def evaluate(
    dist_fn: Callable[[chess.Board], tuple[dict[str, float], str]],
    games: Iterable[Game],
    max_games: int | None = None,
    skip: Mapping[str, Collection[int]] | None = None,
) -> list[MoveResult]:
    """Score `dist_fn` on Justin's moves. `max_games` takes a seeded random sample."""
    games = list(games)
    if max_games is not None and max_games < len(games):
        games = random.Random(0).sample(games, max_games)
    results = []
    for game, board, played in justin_positions(games, skip):
        dist, source = dist_fn(board)
        ranked = sorted(dist, key=lambda m: -dist[m])
        results.append(
            MoveResult(
                top1=bool(ranked) and ranked[0] == played,
                top3=played in ranked[:3],
                logprob=math.log(max(dist.get(played, 0.0), LOGPROB_FLOOR)),
                phase=classify_phase(board),
                time_class=game.time_class,
                source=source,
            )
        )
    return results


def summarize(results: Sequence[MoveResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[MoveResult]] = {"all": list(results)}
    for r in results:
        groups.setdefault(f"phase:{r.phase}", []).append(r)
        groups.setdefault(f"time:{r.time_class}", []).append(r)
        groups.setdefault(f"source:{r.source}", []).append(r)
    return {
        key: {
            "n": len(rs),
            "top1": sum(r.top1 for r in rs) / len(rs),
            "top3": sum(r.top3 for r in rs) / len(rs),
            "mean_logprob": sum(r.logprob for r in rs) / len(rs),
        }
        for key, rs in groups.items()
    }
