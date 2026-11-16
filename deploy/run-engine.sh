#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$repo_dir/.venv/bin/python" -m chessbot.uci --config "$repo_dir/config/final.json"
