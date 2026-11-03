import chess

from chessbot.calibrate import calibrate_elo
from chessbot.games import Game


def _game(moves, color=chess.WHITE, i=0) -> Game:
    return Game(url=f"u{i}", end_time=i, time_class="blitz", my_color=color,
                my_elo=1500, opp_elo=1500, moves=tuple(moves))


class PeakedMaia:
    """Gives Justin's move e2e4 its highest probability at Elo 1400."""

    def __init__(self):
        self.seen: list[tuple[str, int]] = []

    def move_probs(self, board, self_elo, oppo_elo):
        self.seen.append((board.fen(), self_elo))
        p = 0.9 - abs(self_elo - 1400) / 2000
        return {"e2e4": p, "d2d4": 1 - p}


def test_picks_elo_with_best_mean_logprob():
    games = [_game(["e2e4"], i=i) for i in range(5)]
    best, scores = calibrate_elo(PeakedMaia(), games, candidates=range(1000, 1801, 100))
    assert best == 1400
    assert scores[1400] > scores[1000] and scores[1400] > scores[1800]


def test_only_uses_games_passed_in():
    passed = [_game(["e2e4", "e7e5", "g1f3"], chess.WHITE, i=i) for i in range(3)]
    maia = PeakedMaia()
    calibrate_elo(maia, passed, candidates=[1400])
    expected = set()
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        if board.turn == chess.WHITE:
            expected.add(board.fen())
        board.push_uci(uci)
    assert {fen for fen, _ in maia.seen} == expected


def test_sampling_is_seeded():
    firsts = ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "f2f4"]
    games = [_game([m], i=i) for i, m in enumerate(firsts)]
    runs = []
    for seed in (0, 0):
        maia = PeakedMaia()
        calibrate_elo(maia, games, candidates=[1400], sample_games=2, seed=seed)
        runs.append(maia.seen)
    assert runs[0] == runs[1]
    assert len(runs[0]) == 2
