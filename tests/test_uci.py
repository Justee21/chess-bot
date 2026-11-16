import io
import random

import chess
import pytest

from chessbot.policy import Choice
from chessbot.uci import parse_position, run


class RecordingPolicy:
    def __init__(self):
        self.boards = []

    def choose(self, board, rng):
        self.boards.append(board.copy())
        return Choice(next(iter(board.legal_moves)), "maia")


def test_position_replays_history_needed_by_maia_and_draw_detection():
    board = parse_position("startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split())
    assert len(board.move_stack) == 7
    board.push_uci("f6g8")
    assert board.is_repetition(3)


def test_fen_history_and_illegal_move_handling():
    board = chess.Board()
    board.push_uci("e2e4")
    rebuilt = parse_position(["fen", *board.fen().split(), "moves", "e7e5"])
    board.push_uci("e7e5")
    assert rebuilt.fen() == board.fen()
    with pytest.raises(ValueError):
        parse_position("startpos moves e2e5".split())


def test_protocol_answers_handshake_and_resets_between_games():
    source = io.StringIO("uci\nisready\nposition startpos moves e2e4\ngo wtime 10000 btime 10000\nucinewgame\ngo\nquit\n")
    sink, policy = io.StringIO(), RecordingPolicy()
    run(policy, source, sink, random.Random(0))
    lines = sink.getvalue().splitlines()
    assert "uciok" in lines and "readyok" in lines
    assert len([line for line in lines if line.startswith("bestmove ")]) == 2
    assert policy.boards[0].turn == chess.BLACK
    assert policy.boards[1].fen() == chess.STARTING_FEN


def test_terminal_position_returns_null_move_without_calling_policy():
    policy, sink = RecordingPolicy(), io.StringIO()
    run(policy, io.StringIO("position startpos moves f2f3 e7e5 g2g4 d8h4\ngo\n"), sink, random.Random(0))
    assert sink.getvalue() == "bestmove 0000\n"
    assert policy.boards == []


def test_invalid_position_does_not_silently_play_a_stale_board():
    with pytest.raises(ValueError):
        run(RecordingPolicy(), io.StringIO("position startpos moves e2e5\ngo\n"), io.StringIO(), random.Random(0))
