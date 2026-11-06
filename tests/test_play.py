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


def test_choose_color_honors_explicit_colors_and_randomizes_only_on_random():
    from chessbot.play import choose_color

    for seed in range(10):
        assert choose_color("black", random.Random(seed)) == chess.BLACK
        assert choose_color("white", random.Random(seed)) == chess.WHITE
    assert {choose_color("random", random.Random(seed)) for seed in range(20)} == {chess.WHITE, chess.BLACK}


def test_pick_blind_is_seeded_and_covers_every_option():
    from chessbot.play import pick_blind

    names = ["base", "ft1", "ft2"]
    assert pick_blind(names, random.Random(3)) == pick_blind(names, random.Random(3))
    assert {pick_blind(names, random.Random(seed)) for seed in range(40)} == set(names)


def test_log_blind_appends_one_json_line_per_game(tmp_path):
    import json

    from chessbot.play import log_blind

    path = tmp_path / "sub" / "blind.jsonl"
    log_blind(path, {"actual": "ft2", "guess": "base", "result": "1-0"})
    log_blind(path, {"actual": "base", "guess": "base", "result": "*"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["actual"] for r in rows] == ["ft2", "base"]


def test_tally_judgments_counts_yes_per_model_and_ignores_unfinished_rows():
    from chessbot.play import tally_judgments

    rows = [
        {"actual": "ft2", "like_me": True},
        {"actual": "ft2", "like_me": False},
        {"actual": "ft2", "like_me": True},
        {"actual": "base", "like_me": False},
        {"actual": "ft1"},
    ]
    assert tally_judgments(rows, ["base", "ft1", "ft2"]) == {
        "base": (0, 1), "ft1": (0, 0), "ft2": (2, 3),
    }


def test_load_blind_rows_skips_missing_file_and_blank_lines(tmp_path):
    from chessbot.play import load_blind_rows, log_blind

    path = tmp_path / "blind.jsonl"
    assert load_blind_rows(path) == []
    log_blind(path, {"actual": "ft1", "like_me": True})
    path.write_text(path.read_text() + "\n")
    assert load_blind_rows(path) == [{"actual": "ft1", "like_me": True}]


def test_y_or_n_mid_game_ends_it_with_a_verdict_only_when_judging():
    from chessbot.play import EarlyVerdict

    policy = HarnessPolicy(None, StubMaia({}), elo=1500, temperature=1.0)
    for key, expected in [("y", True), ("N", False)]:
        typed = iter(["e4", key])
        with pytest.raises(EarlyVerdict) as caught:
            play_game_vs_human(
                policy, human=chess.WHITE, rng=random.Random(0),
                read=lambda prompt: next(typed), write=lambda line: None, judging=True,
            )
        assert caught.value.like_me is expected
    typed = iter(["y", "quit"])
    lines: list[str] = []
    play_game_vs_human(
        policy, human=chess.WHITE, rng=random.Random(0),
        read=lambda prompt: next(typed), write=lines.append,
    )
    assert any("not a legal move" in line.lower() for line in lines)
