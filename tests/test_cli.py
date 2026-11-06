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


def _rated_raw_dir(tmp_path, n=40):
    """Game i is rated 1000+i for Justin, so an example's self_elo tells which game it came from."""
    raw = tmp_path / "raw_rated"
    raw.mkdir()
    games = [
        {
            "url": f"u{i}",
            "pgn": '[White "Justin0921"]\n[Black "x"]\n\n1. e4 e5 *\n',
            "time_class": "blitz",
            "rules": "chess",
            "end_time": 1000 + i,
            "white": {"username": "Justin0921", "rating": 1000 + i},
            "black": {"username": "x", "rating": 1500},
        }
        for i in range(n)
    ]
    (raw / "2023-11.json").write_text(json.dumps({"games": games}))
    return raw


def _export(tmp_path):
    from maia3.utils import get_all_possible_moves

    from chessbot.cli import export_examples
    from chessbot.examples import load_examples

    out = tmp_path / "ft"
    move_index = {m: i for i, m in enumerate(get_all_possible_moves())}
    sizes = export_examples(_rated_raw_dir(tmp_path), "Justin0921", out, move_index)
    return sizes, load_examples(out / "train.npz"), load_examples(out / "val.npz")


def test_export_examples_never_writes_test_games(tmp_path):
    sizes, train, val = _export(tmp_path)
    assert sizes == (31, 3, 6)
    test_ratings = set(range(1034, 1040))
    assert not {e.self_elo for e in train + val} & test_ratings
    assert len(train) == 31 and len(val) == 3  # one Justin move per game


def test_export_validation_games_are_newer_than_training_games(tmp_path):
    _, train, val = _export(tmp_path)
    assert max(e.self_elo for e in train) < min(e.self_elo for e in val)


def test_finetune_refuses_to_overwrite_existing_checkpoint(tmp_path):
    import pytest

    from chessbot.cli import main

    existing = tmp_path / "ft.pt"
    existing.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["finetune", "--examples", str(tmp_path / "nowhere"), "--out", str(existing)])
    assert existing.read_bytes() == b"x"


def test_export_examples_leaves_out_scramble_moves(tmp_path):
    from maia3.utils import get_all_possible_moves

    from chessbot.cli import export_examples
    from chessbot.examples import load_examples

    move_index = {m: i for i, m in enumerate(get_all_possible_moves())}
    skip = {"u0": {0}, "u1": {0}}  # the only White move of two training games
    export_examples(_rated_raw_dir(tmp_path), "Justin0921", tmp_path / "ft", move_index, skip=skip)
    train = load_examples(tmp_path / "ft" / "train.npz")
    assert len(train) == 31 - 2
    assert not {1000, 1001} & {e.self_elo for e in train}
