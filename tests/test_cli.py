import json

from chessbot.cli import _split, profile_sample


def _raw_dir(tmp_path, n=40):
    raw = tmp_path / "raw"
    raw.mkdir()
    games = [
        {
            "url": f"u{i}",
            "pgn": '[White "Justin0921"]\n[Black "x"]\n\n1. e4 e5 *\n',
            "time_class": "blitz",
            "rules": "chess",
            "end_time": 1000 + i,
            "white": {"username": "Justin0921", "rating": 1500},
            "black": {"username": "x", "rating": 1500},
        }
        for i in range(n)
    ]
    (raw / "2023-11.json").write_text(json.dumps({"games": games}))
    return raw


def test_train_profile_sample_never_contains_test_games(tmp_path):
    raw = _raw_dir(tmp_path)
    _, test = _split(raw, "Justin0921")
    sample = profile_sample(raw, "Justin0921", n=1000, split="train")
    assert sample
    assert not {g.url for g in sample} & {g.url for g in test}


def test_test_profile_sample_contains_only_test_games(tmp_path):
    raw = _raw_dir(tmp_path)
    _, test = _split(raw, "Justin0921")
    sample = profile_sample(raw, "Justin0921", n=1000, split="test")
    assert {g.url for g in sample} == {g.url for g in test}
