"""Ingest chess.com monthly archives into Game records."""
from __future__ import annotations

import io
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

USER_AGENT = "chessbot/0.1 (personal research project)"


@dataclass(frozen=True)
class Game:
    url: str
    end_time: int  # unix seconds
    time_class: str  # "bullet" | "blitz" | "rapid" | "daily"
    my_color: chess.Color
    my_elo: int
    opp_elo: int
    moves: tuple[str, ...]  # whole game, UCI


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def download_archives(username: str, out_dir: Path, delay: float = 0.3) -> int:
    """Write one YYYY-MM.json per monthly archive; return the number written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = _get_json(f"https://api.chess.com/pub/player/{username}/games/archives")["archives"]
    for url in archives:
        year, month = url.rstrip("/").split("/")[-2:]
        (out_dir / f"{year}-{month}.json").write_text(json.dumps(_get_json(url)))
        time.sleep(delay)
    return len(archives)


def _parse_moves(pgn: str) -> tuple[str, ...]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None or game.errors:
        return ()
    return tuple(move.uci() for move in game.mainline_moves())


def load_games(raw_dir: Path, username: str = "Justin0921") -> list[Game]:
    """Standard-chess games for `username`, oldest first."""
    games: list[Game] = []
    for path in sorted(raw_dir.glob("*.json")):
        for raw in json.loads(path.read_text()).get("games", []):
            if raw.get("rules") != "chess" or not raw.get("pgn"):
                continue
            white, black = raw["white"], raw["black"]
            if white["username"].lower() == username.lower():
                color, me, opp = chess.WHITE, white, black
            elif black["username"].lower() == username.lower():
                color, me, opp = chess.BLACK, black, white
            else:
                continue
            moves = _parse_moves(raw["pgn"])
            if not moves:
                continue
            games.append(
                Game(
                    url=raw["url"],
                    end_time=raw["end_time"],
                    time_class=raw["time_class"],
                    my_color=color,
                    my_elo=me["rating"],
                    opp_elo=opp["rating"],
                    moves=moves,
                )
            )
    games.sort(key=lambda g: g.end_time)
    return games
