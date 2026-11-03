import chess
import torch
import pytest

from chessbot.maia import MaiaPredictor, history_frames

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def maia():
    return MaiaPredictor("maia3-5m")


def _legal(board: chess.Board) -> set[str]:
    return {m.uci() for m in board.legal_moves}


def _top(probs: dict[str, float], n: int) -> list[str]:
    return sorted(probs, key=lambda m: -probs[m])[:n]


def test_start_position_probs_sum_to_one_over_legal_moves(maia):
    board = chess.Board()
    probs = maia.move_probs(board, 1500, 1500)
    assert set(probs) == _legal(board)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_start_position_prefers_common_openings(maia):
    probs = maia.move_probs(chess.Board(), 1500, 1500)
    assert {"e2e4", "d2d4"} & set(_top(probs, 3))


def test_black_to_move_returns_black_moves(maia):
    board = chess.Board()
    board.push_uci("e2e4")
    probs = maia.move_probs(board, 1500, 1500)
    assert set(probs) == _legal(board)
    assert {"e7e5", "c7c5"} & set(_top(probs, 3))


def test_promotion_moves_survive_mirroring(maia):
    white = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    probs = maia.move_probs(white, 1500, 1500)
    assert {"a7a8q", "a7a8n"} <= set(probs)
    black = chess.Board("7K/8/8/8/8/8/p7/7k b - - 0 1")
    probs = maia.move_probs(black, 1500, 1500)
    assert {"a2a1q", "a2a1n"} <= set(probs)


def test_single_legal_move_gets_probability_one(maia):
    board = chess.Board("7k/5K2/8/8/8/8/8/8 b - - 0 1")
    assert _legal(board) == {"h8h7"}
    probs = maia.move_probs(board, 1500, 1500)
    assert probs == pytest.approx({"h8h7": 1.0})


def test_elo_changes_distribution(maia):
    board = chess.Board()
    low = maia.move_probs(board, 800, 800)
    high = maia.move_probs(board, 2200, 2200)
    assert low != pytest.approx(high)


def test_checkpoint_argument_loads_that_file(maia, tmp_path):
    state = {k: v.clone() for k, v in maia.model.state_dict().items()}
    key = next(k for k in state if k.startswith("proj_sq_from") and k.endswith("weight"))
    state[key] = state[key] * 1.5
    path = tmp_path / "ft.pt"
    torch.save({"model_state_dict": state}, path)
    tuned = MaiaPredictor("maia3-5m", checkpoint=path)
    board = chess.Board()
    base = maia.move_probs(board, 1500, 1500)
    changed = tuned.move_probs(board, 1500, 1500)
    assert any(abs(base[m] - changed[m]) > 1e-6 for m in base)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MaiaPredictor("maia3-5m", checkpoint=tmp_path / "nope.pt")


def test_history_frames_matches_predictor_history(maia):
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6", "b1d2", "c8e6"):
        board.push_uci(uci)
        frames = history_frames(board, 8)
        expected = maia._history(board)
        assert len(frames) == len(expected)
        assert all(torch.equal(a, b) for a, b in zip(frames, expected))
