import random

import chess

from chessbot.policy import Choice
from chessbot.selfplay import pick_temperature, play_game


class ScriptedPolicy:
    def __init__(self, script):
        self.script = script

    def choose(self, board, rng):
        move = chess.Move.from_uci(self.script[len(board.move_stack) % len(self.script)])
        return Choice(move, "maia")


class RandomPolicy:
    def choose(self, board, rng):
        return Choice(rng.choice(sorted(board.legal_moves, key=str)), "maia")


def test_play_game_ends_on_checkmate():
    policy = ScriptedPolicy(["f2f3", "e7e5", "g2g4", "d8h4"])  # fool's mate
    assert play_game(policy, random.Random(0)) == ("f2f3", "e7e5", "g2g4", "d8h4")


def test_play_game_respects_max_plies():
    policy = ScriptedPolicy(["g1f3", "g8f6", "f3g1", "f6g8"])
    assert len(play_game(policy, random.Random(0), max_plies=4)) == 4


def test_play_game_stops_at_claimable_draw():
    policy = ScriptedPolicy(["g1f3", "g8f6", "f3g1", "f6g8"])
    moves = play_game(policy, random.Random(0), max_plies=400)
    assert len(moves) < 400  # threefold repetition ends it


def test_play_game_is_seeded():
    a = play_game(RandomPolicy(), random.Random(1), max_plies=40)
    b = play_game(RandomPolicy(), random.Random(1), max_plies=40)
    c = play_game(RandomPolicy(), random.Random(2), max_plies=40)
    assert a == b
    assert a != c


def test_pick_temperature_chooses_closest_avg_cpl():
    profiles = {
        0.6: {"avg_cpl": 40.0}, 0.8: {"avg_cpl": 60.0},
        1.0: {"avg_cpl": 85.0}, 1.2: {"avg_cpl": 120.0},
    }
    assert pick_temperature(profiles, {"avg_cpl": 80.0}) == 1.0
