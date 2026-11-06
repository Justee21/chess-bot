from types import SimpleNamespace

import chess
import numpy as np
import pytest
import torch
from maia3.dataset import get_historical_tokens
from maia3.utils import get_all_possible_moves, mirror_move

from chessbot.examples import build_examples, collate, expand_frames, load_examples, save_examples
from chessbot.games import Game
from chessbot.maia import history_frames

VOCAB = get_all_possible_moves()
MOVE_INDEX = {m: i for i, m in enumerate(VOCAB)}
CFG = SimpleNamespace(history=8, include_time_info=False)

OPENING = ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6")
PROMOTION = ("h2h4", "g7g5", "h4g5", "h7h6", "g5h6", "g8f6", "h6h7", "f6g8", "h7g8q")
FORCED_KING_MOVE = ("e2e3", "b7b6", "g2g4", "g7g5", "g1h3", "h7h5", "d1f3", "b6b5", "f3f7", "e8f7")  # Kxf7 is the only legal reply


def _game(moves, color=chess.WHITE, time_class="blitz", my_elo=1500, opp_elo=1600) -> Game:
    return Game(url="u", end_time=1, time_class=time_class, my_color=color,
                my_elo=my_elo, opp_elo=opp_elo, moves=tuple(moves))


def test_one_example_per_move_by_justin_only():
    white = build_examples([_game(OPENING)], MOVE_INDEX)
    black = build_examples([_game(OPENING, chess.BLACK)], MOVE_INDEX)
    assert [e.target for e in white] == [MOVE_INDEX[m] for m in ("e2e4", "g1f3", "f1c4")]
    assert [e.target for e in black] == [MOVE_INDEX[mirror_move(m)] for m in ("e7e5", "b8c6", "g8f6")]


def test_ratings_are_justin_then_opponent():
    (first, *_) = build_examples([_game(OPENING, chess.BLACK, my_elo=1234, opp_elo=1777)], MOVE_INDEX)
    assert (first.self_elo, first.oppo_elo) == (1234, 1777)


def test_black_target_and_legal_indices_are_mirrored():
    (ex,) = build_examples([_game(("e2e4", "e7e5"), chess.BLACK)], MOVE_INDEX)
    assert ex.target == MOVE_INDEX[mirror_move("e7e5")]
    assert ex.target in ex.legal
    assert len(ex.legal) == 20


def test_promotion_target_is_valid_and_legal():
    examples = build_examples([_game(PROMOTION)], MOVE_INDEX)
    last = examples[-1]
    assert VOCAB[last.target] == "h7g8q"
    assert last.target in last.legal and last.target < len(VOCAB)


def test_single_legal_move_position_still_yields_an_example():
    examples = build_examples([_game(FORCED_KING_MOVE, chess.BLACK)], MOVE_INDEX)
    assert len(examples) == 5
    assert len(examples[-1].legal) == 1
    assert examples[-1].legal[0] == examples[-1].target


def test_bullet_moves_are_down_weighted():
    bullet = build_examples([_game(OPENING, time_class="bullet")], MOVE_INDEX, bullet_weight=0.25)
    blitz = build_examples([_game(OPENING, time_class="blitz")], MOVE_INDEX, bullet_weight=0.25)
    assert {e.weight for e in bullet} == {0.25}
    assert {e.weight for e in blitz} == {1.0}


def _boards_before_justin_moves(game):
    board, out = chess.Board(), []
    for uci in game.moves:
        if board.turn == game.my_color:
            out.append(board.copy())
        board.push_uci(uci)
    return out


@pytest.mark.parametrize("moves,color", [(OPENING, chess.WHITE), (OPENING, chess.BLACK), (PROMOTION, chess.WHITE)])
def test_expand_frames_matches_get_historical_tokens(moves, color):
    game = _game(moves, color)
    examples = build_examples([game], MOVE_INDEX)
    boards = _boards_before_justin_moves(game)
    assert len(examples) == len(boards)
    expanded = expand_frames(np.stack([e.frames for e in examples]))
    for i, board in enumerate(boards):
        expected = get_historical_tokens(history_frames(board, 8), CFG, 0.0, 0.0, 0.0, 0.0)[:, :96]
        assert torch.equal(expanded[i], expected)


def test_save_load_round_trip(tmp_path):
    examples = build_examples([_game(OPENING), _game(FORCED_KING_MOVE, chess.BLACK, "bullet")], MOVE_INDEX)
    path = tmp_path / "ex.npz"
    save_examples(examples, path)
    loaded = load_examples(path)
    assert len(loaded) == len(examples)
    for a, b in zip(examples, loaded):
        assert np.array_equal(a.frames, b.frames) and np.array_equal(a.legal, b.legal)
        assert (a.self_elo, a.oppo_elo, a.target, a.weight) == (b.self_elo, b.oppo_elo, b.target, b.weight)


def test_collate_masks_exactly_the_legal_moves():
    examples = build_examples([_game(OPENING), _game(FORCED_KING_MOVE, chess.BLACK)], MOVE_INDEX)
    tokens, self_elo, oppo_elo, target, mask, weight = collate(examples, len(VOCAB))
    assert tokens.shape == (len(examples), 64, 96)
    assert mask.shape == (len(examples), len(VOCAB)) and mask.dtype == torch.bool
    for i, e in enumerate(examples):
        assert sorted(torch.nonzero(mask[i]).flatten().tolist()) == sorted(e.legal.tolist())
        assert target[i].item() == e.target
    assert self_elo.dtype == oppo_elo.dtype == torch.long
    assert weight.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
def test_training_encoding_gives_the_same_probabilities_as_the_predictor(color):
    from chessbot.maia import MaiaPredictor

    predictor = MaiaPredictor("maia3-5m")
    game = _game(OPENING + ("d2d3", "f8c5", "c2c3", "d7d6", "b1d2", "c8e6"), color)
    examples = build_examples([game], predictor.move_index)
    boards = _boards_before_justin_moves(game)
    tokens, self_elo, oppo_elo, _, mask, _ = collate(examples, len(predictor.all_moves))
    with torch.no_grad():
        logits = predictor.model(tokens, self_elo, oppo_elo)[0].float()
    probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    for i, board in enumerate(boards):
        expected = predictor.move_probs(board, 1500, 1600)
        for uci, p in expected.items():
            idx = predictor.move_index[mirror_move(uci) if board.turn == chess.BLACK else uci]
            assert probs[i, idx].item() == pytest.approx(p, abs=1e-5)


def test_loaded_examples_own_their_arrays_instead_of_viewing_the_whole_file(tmp_path):
    # Views would pin the full stacked arrays once per example (tens of GB on the real data).
    examples = build_examples([_game(OPENING), _game(OPENING, chess.BLACK)], MOVE_INDEX)
    path = tmp_path / "ex.npz"
    save_examples(examples, path)
    for e in load_examples(path):
        assert e.frames.flags.owndata and e.legal.flags.owndata


def test_skipped_plies_produce_no_examples():
    kept = build_examples([_game(OPENING)], MOVE_INDEX, skip={"u": {2}})
    assert [e.target for e in kept] == [MOVE_INDEX[m] for m in ("e2e4", "f1c4")]
    assert len(build_examples([_game(OPENING)], MOVE_INDEX, skip={"other": {2}})) == 3
