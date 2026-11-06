"""Training examples for the fine-tune: one per move Justin played, in Maia's input format."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

import chess
import numpy as np
import torch
import torch.nn.functional as F
from maia3.dataset import tokenize_board
from maia3.utils import mirror_move

from chessbot.games import Game


@dataclass(frozen=True)
class Example:
    frames: np.ndarray  # (history, 64) int8 piece codes, oldest first, padded with the earliest frame
    self_elo: int
    oppo_elo: int
    target: int  # vocabulary index of the move Justin played (mirrored when he is Black)
    legal: np.ndarray  # int16 vocabulary indices of every legal move (mirrored for Black)
    weight: float


def _compress(tokens: torch.Tensor) -> np.ndarray:
    """(64, 12) one-hot piece planes to (64,) int8 codes, 0 = empty."""
    return ((tokens.argmax(dim=1) + 1) * tokens.any(dim=1)).to(torch.int8).numpy()


def build_examples(
    games: Iterable[Game],
    move_index: dict[str, int],
    bullet_weight: float = 0.5,
    history: int = 8,
    skip: Mapping[str, Collection[int]] | None = None,
) -> list[Example]:
    """One example per move Justin made; moves outside Maia's vocabulary are skipped, and so are
    the plies listed for a game's url in `skip`."""
    examples: list[Example] = []
    for game in games:
        board = chess.Board()
        frames: deque = deque([_compress(tokenize_board(board))], maxlen=history)
        weight = bullet_weight if game.time_class == "bullet" else 1.0
        skipped = (skip or {}).get(game.url, ())
        for ply, uci in enumerate(game.moves):
            if board.turn == game.my_color and ply not in skipped:
                black = board.turn == chess.BLACK
                played = mirror_move(uci) if black else uci
                if played in move_index:
                    legal = [
                        move_index[u]
                        for u in ((mirror_move(m.uci()) if black else m.uci()) for m in board.legal_moves)
                        if u in move_index
                    ]
                    padded = [frames[0]] * (history - len(frames)) + list(frames)
                    examples.append(Example(
                        frames=np.stack(padded), self_elo=game.my_elo, oppo_elo=game.opp_elo,
                        target=move_index[played], legal=np.array(legal, dtype=np.int16), weight=weight,
                    ))
            board.push_uci(uci)
            frames.append(_compress(tokenize_board(board)))
    return examples


def expand_frames(frames: np.ndarray) -> torch.Tensor:
    """(B, history, 64) int8 codes to the model's (B, 64, 12 * history) float input."""
    codes = torch.as_tensor(frames).long()
    planes = F.one_hot(codes, 13)[..., 1:].float()  # (B, history, 64, 12)
    batch, history = planes.shape[0], planes.shape[1]
    return planes.permute(0, 2, 1, 3).reshape(batch, 64, history * 12)


def collate(examples: Sequence[Example], vocab_size: int) -> tuple[torch.Tensor, ...]:
    """(tokens, self_elo, oppo_elo, target, legal_mask, weight) for a batch."""
    tokens = expand_frames(np.stack([e.frames for e in examples]))
    mask = torch.zeros((len(examples), vocab_size), dtype=torch.bool)
    for i, e in enumerate(examples):
        mask[i, torch.as_tensor(e.legal).long()] = True
    return (
        tokens,
        torch.tensor([e.self_elo for e in examples], dtype=torch.long),
        torch.tensor([e.oppo_elo for e in examples], dtype=torch.long),
        torch.tensor([e.target for e in examples], dtype=torch.long),
        mask,
        torch.tensor([e.weight for e in examples], dtype=torch.float32),
    )


def save_examples(examples: Sequence[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(e.legal) for e in examples]).astype(np.int64)
    np.savez_compressed(
        path,
        frames=np.stack([e.frames for e in examples]) if examples else np.zeros((0, 8, 64), dtype=np.int8),
        self_elo=np.array([e.self_elo for e in examples], dtype=np.int32),
        oppo_elo=np.array([e.oppo_elo for e in examples], dtype=np.int32),
        target=np.array([e.target for e in examples], dtype=np.int32),
        weight=np.array([e.weight for e in examples], dtype=np.float64),
        legal=np.concatenate([e.legal for e in examples]) if examples else np.zeros(0, dtype=np.int16),
        offsets=offsets,
    )


def load_examples(path: Path) -> list[Example]:
    """Read each array once and copy per-example slices, so no example pins the whole file in memory."""
    with np.load(path) as data:
        frames, legal, offsets = data["frames"], data["legal"], data["offsets"]
        self_elo, oppo_elo = data["self_elo"], data["oppo_elo"]
        target, weight = data["target"], data["weight"]
    return [
        Example(
            frames=frames[i].copy(), self_elo=int(self_elo[i]), oppo_elo=int(oppo_elo[i]),
            target=int(target[i]), legal=legal[offsets[i]:offsets[i + 1]].astype(np.int16),
            weight=float(weight[i]),
        )
        for i in range(len(target))
    ]
