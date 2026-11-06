import json
import shutil
from pathlib import Path

import chess

from chessbot.games import Game, load_games

FIXTURES = Path(__file__).parent / "fixtures"


def _raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURES / "archive_sample.json", raw / "2023-11.json")
    return raw


def test_load_games_keeps_only_standard_chess(tmp_path):
    games = load_games(_raw_dir(tmp_path))
    assert len(games) == 2


def test_load_games_sorted_by_end_time(tmp_path):
    games = load_games(_raw_dir(tmp_path))
    assert [g.end_time for g in games] == [1700000100, 1700000200]


def test_my_color_and_elos_follow_username(tmp_path):
    black_game, white_game = load_games(_raw_dir(tmp_path))
    assert white_game.my_color == chess.WHITE
    assert (white_game.my_elo, white_game.opp_elo) == (1500, 1480)
    assert black_game.my_color == chess.BLACK
    assert (black_game.my_elo, black_game.opp_elo) == (1450, 1620)
    assert black_game.time_class == "bullet"


def test_moves_are_uci_tuples(tmp_path):
    _, white_game = load_games(_raw_dir(tmp_path))
    assert white_game.moves == ("e2e4", "e7e5", "g1f3", "b8c6")
    assert isinstance(white_game, Game)


def test_game_from_custom_start_position_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pgn = (
        '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]\n'
        '[White "Justin0921"]\n[Black "x"]\n\n1. e4 Kd7 *\n'
    )
    odd = {
        "games": [
            {
                "url": "u", "pgn": pgn, "time_class": "blitz", "rules": "chess",
                "end_time": 1,
                "white": {"username": "Justin0921", "rating": 1500},
                "black": {"username": "x", "rating": 1500},
            }
        ]
    }
    (raw / "2023-11.json").write_text(json.dumps(odd))
    assert load_games(raw) == []


def test_game_with_unparseable_pgn_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    bad = {
        "games": [
            {
                "url": "u",
                "pgn": "this is not a pgn",
                "time_class": "blitz",
                "rules": "chess",
                "end_time": 1,
                "white": {"username": "Justin0921", "rating": 1500},
                "black": {"username": "x", "rating": 1500},
            }
        ]
    }
    (raw / "2023-11.json").write_text(json.dumps(bad))
    assert load_games(raw) == []


def _one_game_archive(tmp_path: Path, pgn: str) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    game = {
        "url": "u", "pgn": pgn, "time_class": "bullet", "rules": "chess", "end_time": 1,
        "white": {"username": "Justin0921", "rating": 1500},
        "black": {"username": "x", "rating": 1500},
    }
    (raw / "2023-11.json").write_text(json.dumps({"games": [game]}))
    return raw


def test_clocks_are_seconds_left_after_each_move(tmp_path):
    pgn = ('[White "Justin0921"]\n[Black "x"]\n\n'
           '1. e4 {[%clk 0:00:58.5]} 1... e5 {[%clk 0:00:57]} 2. Nf3 {[%clk 0:01:03.2]} *\n')
    (game,) = load_games(_one_game_archive(tmp_path, pgn))
    assert game.clocks == (58.5, 57.0, 63.2)


def test_game_without_clocks_has_empty_clocks(tmp_path):
    pgn = '[White "Justin0921"]\n[Black "x"]\n\n1. e4 e5 2. Nf3 *\n'
    (game,) = load_games(_one_game_archive(tmp_path, pgn))
    assert game.clocks == ()
