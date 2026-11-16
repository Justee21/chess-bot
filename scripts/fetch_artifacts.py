"""Download the published checkpoint and table, verifying the frozen SHA-256 hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from chessbot.frozen import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Justee21/chess-bot")
    parser.add_argument("--release", default="v0.1.0-frozen")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config/final.json").read_text())
    for key in ("checkpoint", "table"):
        dest = root / cfg[key]
        expected = cfg[f"{key}_sha256"]
        if dest.exists():
            if sha256(dest) != expected:
                raise SystemExit(f"{dest} has a different hash; refusing to overwrite it")
            print(f"verified {dest.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{args.repo}/releases/download/{args.release}/{dest.name}"
        request = urllib.request.Request(url, headers={"User-Agent": "chessbot/0.1"})
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha256(temporary) != expected:
                raise SystemExit(f"SHA-256 mismatch downloading {dest.name}")
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"downloaded and verified {dest.name}")


if __name__ == "__main__":
    main()
