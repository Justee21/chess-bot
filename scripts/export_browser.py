"""Export the frozen FP32 Maia policy for static browser hosting and check parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch

from chessbot.frozen import read_config, verify_artifacts
from chessbot.maia import MaiaPredictor
from chessbot.policy import HarnessPolicy
from chessbot.table import MoveTable
from maia3.dataset import get_historical_tokens


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FrozenModel(torch.nn.Module):
    def __init__(self, model, elo: int):
        super().__init__()
        self.model = model
        self.register_buffer("elo", torch.tensor([elo], dtype=torch.long))

    def forward(self, tokens):
        return self.model(tokens, self.elo, self.elo)[0]


def positions(count: int) -> list[chess.Board]:
    result = [chess.Board()]
    for fen in (
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "k7/8/1K6/8/8/8/8/2Q5 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 w - - 99 1",
    ):
        result.append(chess.Board(fen))
    repeated = chess.Board()
    for uci in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split():
        repeated.push_uci(uci)
    result.append(repeated)
    rng = random.Random(481)
    while len(result) < count:
        board = chess.Board()
        for _ in range(rng.randint(1, 110)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if any(board.legal_moves):
            result.append(board)
    return result[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/final.json"))
    parser.add_argument("--out", type=Path, default=Path("web"))
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/browser-fixtures.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/onnx-export.json"))
    args = parser.parse_args()
    cfg, root = read_config(args.config)
    verify_artifacts(cfg, root)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    predictor = MaiaPredictor(checkpoint=root / cfg["checkpoint"])
    model = FrozenModel(predictor.model, cfg["elo_input"]).eval()
    channels = 12 * predictor.cfg.history + (3 if predictor.cfg.include_time_info else 0)
    args.out.mkdir(parents=True, exist_ok=True)
    export_path = args.out / "model.onnx"
    sample = torch.zeros((1, 64, channels), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(model, (sample,), str(export_path), input_names=["tokens"],
                          output_names=["logits"], opset_version=18, dynamo=True,
                          external_data=False)
    onnx.checker.check_model(str(export_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(export_path), options, providers=["CPUExecutionProvider"])
    table = MoveTable.load(root / cfg["table"])
    torch_policy = HarnessPolicy(table, predictor, cfg["elo_input"], cfg["temperature"],
                                 min_count=cfg["table_min_count"], min_p=cfg["min_p"])
    max_logit_error = max_probability_error = 0.0
    agreement = 0
    timings = []
    fixtures = []
    for i, board in enumerate(positions(args.positions)):
        tokens = get_historical_tokens(predictor._history(board), predictor.cfg, 0, 0, 0, 0)
        tokens = tokens[:, :channels].unsqueeze(0)
        with torch.no_grad():
            expected = model(tokens).numpy()[0]
        t0 = time.perf_counter()
        actual = session.run(None, {"tokens": tokens.numpy()})[0][0]
        timings.append((time.perf_counter() - t0) * 1000)
        max_logit_error = max(max_logit_error, float(np.max(np.abs(expected - actual))))
        legal = predictor.move_probs(board, cfg["elo_input"], cfg["elo_input"])
        indices = [predictor.move_index[chess.Move.from_uci(m).uci() if board.turn else
                   chess.Move(chess.square_mirror(chess.Move.from_uci(m).from_square),
                              chess.square_mirror(chess.Move.from_uci(m).to_square),
                              promotion=chess.Move.from_uci(m).promotion).uci()] for m in legal]
        logits = actual[indices].astype(np.float64)
        probabilities = np.exp(logits - logits.max()); probabilities /= probabilities.sum()
        expected_probabilities = np.array(list(legal.values()))
        error = float(np.max(np.abs(probabilities - expected_probabilities)))
        max_probability_error = max(max_probability_error, error)
        agreement += int(np.argmax(probabilities) == np.argmax(expected_probabilities))
        if i < 48:
            initial = board.root()
            history = [initial.fen()]
            for move in board.move_stack:
                initial.push(move); history.append(initial.fen())
            dist, source = torch_policy.distribution(board)
            pool = torch_policy._playable(board, dist)
            if not pool and source == "table":
                dist, source = torch_policy._maia_distribution(board), "maia"
                pool = torch_policy._playable(board, dist)
            if not pool:
                pool = torch_policy._playable(board, {m.uci(): 1.0 for m in board.legal_moves}) or dist
            pool = torch_policy._trim(pool)
            total = sum(pool.values())
            fixtures.append({"fen": board.fen(), "initialFen": board.root().fen(),
                             "moves": [m.uci() for m in board.move_stack], "history": history,
                             "tokenOnes": np.flatnonzero(tokens.numpy()).tolist(),
                             "legalProbabilities": legal,
                             "playableProbabilities": {m:p/total for m,p in pool.items()},
                             "source": source})
    if max_probability_error > 1e-5 or agreement != args.positions:
        raise SystemExit(f"parity failed: maximum probability error {max_probability_error}; top1 {agreement}/{args.positions}")
    binary = export_path.read_bytes()
    parts = []
    for i, offset in enumerate(range(0, len(binary), 32 * 1024 * 1024)):
        data = binary[offset:offset + 32 * 1024 * 1024]
        name = f"model-{i+1:02d}.bin"
        (args.out / name).write_bytes(data)
        parts.append({"file": name, "bytes": len(data), "sha256": digest(data)})
    compact_table = {key: dict(value) for key, value in table._positions.items()
                     if sum(value.values()) >= cfg["table_min_count"]}
    table_bytes = json.dumps(compact_table, separators=(",", ":")).encode()
    (args.out / "table.json").write_bytes(table_bytes)
    manifest = {"version": 1, "precision": "float32", "history": predictor.cfg.history,
                "channels": channels, "elo": cfg["elo_input"], "temperature": cfg["temperature"],
                "minP": cfg["min_p"], "tableMinCount": cfg["table_min_count"],
                "checkpointSha256": cfg["checkpoint_sha256"], "sourceTableSha256": cfg["table_sha256"],
                "model": {"bytes": len(binary), "sha256": digest(binary), "parts": parts},
                "table": {"file":"table.json", "bytes":len(table_bytes), "sha256":digest(table_bytes),
                          "positions":len(compact_table)}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures.write_text(json.dumps(fixtures, separators=(",", ":")))
    report = {"positions":args.positions, "top1Agreement":agreement,
              "maxLogitAbsoluteError":max_logit_error, "maxLegalProbabilityAbsoluteError":max_probability_error,
              "nativeOnnxMedianMs":statistics.median(timings), "modelBytes":len(binary),
              "precision":"float32", "quantized":False, "tablePositions":len(compact_table),
              "tableBytes":len(table_bytes), "torch":torch.__version__, "onnx":onnx.__version__,
              "onnxruntime":ort.__version__}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    export_path.unlink()  # static hosting uses small verified chunks, reassembled in the worker
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
