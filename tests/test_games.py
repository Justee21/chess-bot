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
