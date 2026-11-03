import math

import numpy as np
import pytest
import torch
from torch import nn

from chessbot.examples import Example, collate
from chessbot.finetune import masked_loss, save_checkpoint, train

V = 6


class Stub(nn.Module):
    """Same call contract as MAIA3Model: model(tokens, self_elo, oppo_elo)[0] is the logits."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(64 * 96, 32)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 32)])
        self.out = nn.Linear(32, V)

    def forward(self, tokens, self_elo, oppo_elo):
        x = torch.relu(self.head(tokens.flatten(1)))
        for layer in self.transformer.layers:
            x = torch.relu(layer(x))
        return self.out(x), None, None


def _examples(n, seed, learnable=True):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        frames = np.zeros((8, 64), dtype=np.int8)
        code = int(rng.integers(1, 5))
        frames[0, 0] = code
        target = code % 4 if learnable else int(rng.integers(0, 4))
        out.append(Example(frames, 1500, 1500, target, np.arange(V, dtype=np.int16), 1.0))
    return out


def _val_loss(model, examples):
    tokens, se, oe, target, mask, weight = collate(examples, V)
    with torch.no_grad():
        return masked_loss(model(tokens, se, oe)[0], mask, target, torch.ones_like(weight)).item()


def test_masked_loss_ignores_illegal_moves_and_matches_hand_computation():
    mask = torch.tensor([[True, True, False, False]])
    target, weight = torch.tensor([0]), torch.tensor([1.0])
    a = torch.tensor([[0.0, math.log(3), 0.0, 0.0]])
    b = torch.tensor([[0.0, math.log(3), 0.0, 100.0]])
    assert masked_loss(a, mask, target, weight).item() == pytest.approx(math.log(4))
    assert masked_loss(b, mask, target, weight).item() == pytest.approx(math.log(4))


def test_weights_scale_the_loss():
    logits = torch.tensor([[0.0, 0.0], [0.0, math.log(3)]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    target = torch.tensor([0, 0])
    loss = masked_loss(logits, mask, target, torch.tensor([1.0, 0.5]))
    assert loss.item() == pytest.approx((math.log(2) + 0.5 * math.log(4)) / 1.5)


def test_kl_term_is_zero_when_model_equals_base_and_positive_otherwise():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    target, weight = torch.tensor([2]), torch.tensor([1.0])
    plain = masked_loss(logits, mask, target, weight)
    same = masked_loss(logits, mask, target, weight, base_logits=logits.clone(), kl_weight=1.0)
    other = masked_loss(logits, mask, target, weight, base_logits=torch.tensor([[2.0, 1.0, 0.0]]), kl_weight=1.0)
    assert same.item() == pytest.approx(plain.item())
    assert other.item() > plain.item() + 0.1


def test_training_reduces_validation_loss():
    torch.manual_seed(0)
    result = train(Stub(), _examples(200, 0), _examples(100, 1), vocab_size=V,
                   epochs=5, lr=1e-2, batch_size=32, patience=5, log=lambda s: None)
    assert result.history[-1]["val_loss"] < result.history[0]["val_loss"]
    assert result.history[-1]["val_top1"] > result.history[0]["val_top1"]


def test_best_state_is_the_best_epoch_not_the_last():
    torch.manual_seed(0)
    model = Stub()
    val = _examples(100, 2, learnable=False)  # unrelated labels: val loss gets worse as the model fits train
    result = train(model, _examples(200, 0), val, vocab_size=V,
                   epochs=6, lr=1e-1, batch_size=32, patience=10, log=lambda s: None)
    losses = [row["val_loss"] for row in result.history]
    assert result.best_epoch == int(np.argmin(losses))
    assert result.best_epoch < len(losses) - 1
    model.load_state_dict(result.best_state)
    assert _val_loss(model, val) == pytest.approx(min(losses), abs=1e-5)


def test_patience_stops_early():
    torch.manual_seed(0)
    result = train(Stub(), _examples(200, 0), _examples(100, 2, learnable=False), vocab_size=V,
                   epochs=10, lr=1e-1, batch_size=32, patience=1, log=lambda s: None)
    assert len(result.history) < 11  # epoch 0 baseline plus fewer than 10 epochs


def test_freeze_layers_leaves_frozen_weights_unchanged():
    torch.manual_seed(0)
    model = Stub()
    frozen_before = model.transformer.layers[0].weight.detach().clone()
    free_before = model.transformer.layers[1].weight.detach().clone()
    train(model, _examples(100, 0), _examples(50, 1), vocab_size=V, epochs=2, lr=1e-2,
          batch_size=32, freeze_layers=1, patience=5, log=lambda s: None)
    assert torch.equal(model.transformer.layers[0].weight, frozen_before)
    assert not torch.equal(model.transformer.layers[1].weight, free_before)


def test_save_checkpoint_refuses_to_overwrite_and_uses_maia3_format(tmp_path):
    path = tmp_path / "sub" / "ft.pt"
    save_checkpoint(Stub().state_dict(), path)
    assert "model_state_dict" in torch.load(path, weights_only=True)
    with pytest.raises(FileExistsError):
        save_checkpoint(Stub().state_dict(), path)


@pytest.mark.slow
def test_five_m_model_overfits_a_handful_of_positions():
    import chess

    from chessbot.examples import build_examples
    from chessbot.games import Game
    from chessbot.maia import MaiaPredictor

    predictor = MaiaPredictor("maia3-5m")
    moves = ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6",
             "c1e3", "e7e5", "d4b3", "c8e6", "f2f3", "b8d7", "d1d2", "f8e7")
    game = Game("u", 1, "blitz", chess.WHITE, 1500, 1500, moves)
    examples = build_examples([game], predictor.move_index)[:16]
    result = train(predictor.model, examples, examples, vocab_size=len(predictor.all_moves),
                   epochs=30, lr=1e-4, batch_size=16, patience=30, log=lambda s: None)
    assert result.history[-1]["train_loss"] <= 0.7 * result.history[0]["val_loss"]


@pytest.mark.slow
def test_saved_checkpoint_reloads_into_predictor_with_same_probabilities(tmp_path):
    import chess

    from chessbot.examples import build_examples
    from chessbot.games import Game
    from chessbot.maia import MaiaPredictor

    predictor = MaiaPredictor("maia3-5m")
    moves = ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6")
    examples = build_examples([Game("u", 1, "blitz", chess.WHITE, 1500, 1600, moves)], predictor.move_index)
    result = train(predictor.model, examples, examples, vocab_size=len(predictor.all_moves),
                   epochs=2, lr=1e-3, batch_size=4, patience=5, log=lambda s: None)
    path = tmp_path / "ft.pt"
    save_checkpoint(result.best_state, path)
    predictor.model.load_state_dict(result.best_state)
    tokens, se, oe, _, mask, _ = collate(examples[:1], len(predictor.all_moves))
    with torch.no_grad():
        logits = predictor.model(tokens, se, oe)[0].float().masked_fill(~mask, float("-inf"))
    expected = torch.softmax(logits, dim=-1)[0]
    reloaded = MaiaPredictor("maia3-5m", checkpoint=path).move_probs(chess.Board(), 1500, 1600)
    for uci, p in reloaded.items():
        assert expected[predictor.move_index[uci]].item() == pytest.approx(p, abs=1e-5)


def test_history_reports_train_loss_on_the_same_footing_as_validation():
    torch.manual_seed(0)
    model = Stub()
    train_ex, val_ex = _examples(120, 0), _examples(60, 1)
    result = train(model, train_ex, val_ex, vocab_size=V, epochs=1, lr=1e-2,
                   batch_size=32, patience=5, log=lambda s: None)
    row = result.history[-1]
    assert {"train_eval_loss", "train_top1"} <= set(row) and "train_eval_loss" in result.history[0]
    # unweighted, measured after the epoch, on a fixed training subsample as large as validation
    assert row["train_eval_loss"] == pytest.approx(_val_loss(model, train_ex[:len(val_ex)]), abs=1e-5)
