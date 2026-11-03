"""Maia 3 (Chessformer, CSSLab, AGPL-3.0) as a move-probability predictor."""
from __future__ import annotations

from collections import deque

import chess
import torch
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.model_registry import resolve_checkpoint_path
from maia3.uci import load_model, parse_args
from maia3.utils import get_all_possible_moves, mirror_move


class MaiaPredictor:
    def __init__(self, model: str = "maia3-23m", device: str = "cpu"):
        self.cfg = parse_args(["--model", model, "--device", device])
        self.cfg.checkpoint_path = resolve_checkpoint_path(
            self.cfg.model_spec,
            checkpoint_filename=self.cfg.checkpoint_filename,
            cache_dir=self.cfg.cache_dir,
            revision=self.cfg.revision,
            local_files_only=self.cfg.local_files_only,
            force_download=self.cfg.force_download,
            token=self.cfg.hf_token,
        )
        self.model = load_model(self.cfg)
        self.all_moves = get_all_possible_moves()
        self.move_index = {m: i for i, m in enumerate(self.all_moves)}
        self.index_move = {i: m for m, i in self.move_index.items()}

    def _history(self, board: chess.Board) -> deque:
        """Tokenized last `cfg.history` positions, rebuilt from the board's move stack."""
        recent = board.move_stack[-(self.cfg.history - 1):] if self.cfg.history > 1 else []
        start = board.copy()
        for _ in recent:
            start.pop()
        history = deque([tokenize_board(start)], maxlen=self.cfg.history)
        for move in recent:
            start.push(move)
            history.append(tokenize_board(start))
        return history

    @torch.no_grad()
    def move_probs(self, board: chess.Board, self_elo: int, oppo_elo: int) -> dict[str, float]:
        """Probability of each legal move (UCI, real-board orientation); sums to 1."""
        mask = get_legal_moves_mask(board, self.move_index)
        if not bool(mask.any()):
            raise ValueError("no legal moves in this position")
        tokens = get_historical_tokens(
            self._history(board), self.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0
        ).unsqueeze(0)
        logits, _, _ = self.model(
            tokens,
            torch.tensor([self_elo], dtype=torch.long),
            torch.tensor([oppo_elo], dtype=torch.long),
        )
        logits = logits[0].float().masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out = {}
        for idx in torch.nonzero(mask).flatten().tolist():
            uci = self.index_move[idx]
            if board.turn == chess.BLACK:
                uci = mirror_move(uci)
            out[uci] = float(probs[idx])
        return out
