"""Pick the Maia Elo input that best explains Justin's moves (chess.com is not Lichess-scale)."""
from __future__ import annotations

import random
from typing import Sequence

from chessbot.evaluate import evaluate
from chessbot.games import Game
from chessbot.policy import MovePredictor


def calibrate_elo(
    maia: MovePredictor,
    train_games: Sequence[Game],
    candidates: Sequence[int] = range(800, 2201, 100),
    sample_games: int = 150,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """Best Elo by mean log-probability of Justin's moves, using only `train_games`."""
    games = list(train_games)
    if sample_games < len(games):
        games = random.Random(seed).sample(games, sample_games)
    scores: dict[int, float] = {}
    for elo in candidates:
        results = evaluate(lambda board, elo=elo: (maia.move_probs(board, elo, elo), "maia"), games)
        scores[elo] = sum(r.logprob for r in results) / len(results)
    best = max(scores, key=scores.get)
    return best, scores
