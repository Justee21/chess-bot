"""Load the published policy with its original settings and verified artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chessbot.policy import HarnessPolicy
from chessbot.table import MoveTable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    cfg = json.loads(path.read_text())
    if cfg["rules"] != {"resign": False, "offer_or_accept_draws": False, "search": False}:
        raise ValueError("unsupported policy rules in frozen config")
    return cfg, path.parent.parent


def verify_artifacts(cfg: dict, root: Path) -> None:
    for key in ("checkpoint", "table"):
        path = root / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run scripts/fetch_artifacts.py")
        if sha256(path) != cfg[f"{key}_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}; refusing to change the frozen bot")


def load_policy(path: Path) -> HarnessPolicy:
    cfg, root = read_config(path)
    verify_artifacts(cfg, root)
    from chessbot.maia import MaiaPredictor

    return HarnessPolicy(
        MoveTable.load(root / cfg["table"]),
        MaiaPredictor("maia3-23m", checkpoint=root / cfg["checkpoint"]),
        elo=cfg["elo_input"], temperature=cfg["temperature"],
        min_count=cfg["table_min_count"], min_p=cfg["min_p"],
    )
