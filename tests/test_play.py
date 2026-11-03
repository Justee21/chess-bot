import random

import chess
import pytest

from chessbot.play import parse_move, play_game_vs_human
from chessbot.policy import HarnessPolicy


class StubMaia:
    def __init__(self, probs):
        self.probs = probs

    def move_probs(self, board, self_elo, oppo_elo):
        legal = {m.uci() for m in board.legal_moves}
        picked = {m: p for m, p in self.probs.items() if m in legal}
        return picked or {m: 1 / len(legal) for m in legal}


def test_parse_move_accepts_san_and_uci():
    board = chess.Board()
    assert parse_move(board, "e4") == chess.Move.from_uci("e2e4")
    assert parse_move(board, "g1f3") == chess.Move.from_uci("g1f3")
    assert parse_move(board, "  Nf3 ") == chess.Move.from_uci("g1f3")


def test_parse_move_rejects_illegal_or_garbage():
    board = chess.Board()
    for text in ("e5", "zz", "", "Ke2"):
        with pytest.raises(ValueError):
            parse_move(board, text)


def test_human_white_plays_scripted_mate_against_bot():
    # Fool's mate: Justin (Black) after the bot's f3 and g4 mates with Qh4.
    policy = HarnessPolicy(None, StubMaia({"f2f3": 1.0, "g2g4": 1.0}), elo=1500, temperature=1.0)
    typed = iter(["e5", "Qh4#"])
    lines: list[str] = []
    result = play_game_vs_human(
        policy, human=chess.BLACK, rng=random.Random(0),
        read=lambda prompt: next(typed), write=lines.append,
    )
    assert result == "0-1"
    assert any("checkmate" in line.lower() for line in lines)


def test_quit_ends_the_game_without_a_result():
    policy = HarnessPolicy(None, StubMaia({}), elo=1500, temperature=1.0)
    result = play_game_vs_human(
        policy, human=chess.WHITE, rng=random.Random(0),
        read=lambda prompt: "quit", write=lambda line: None,
    )
    assert result == "*"


def test_bad_input_reprompts_instead_of_crashing():
    policy = HarnessPolicy(None, StubMaia({}), elo=1500, temperature=1.0)
    typed = iter(["nonsense", "quit"])
    lines: list[str] = []
    play_game_vs_human(
        policy, human=chess.WHITE, rng=random.Random(0),
        read=lambda prompt: next(typed), write=lines.append,
    )
    assert any("not a legal move" in line.lower() for line in lines)
