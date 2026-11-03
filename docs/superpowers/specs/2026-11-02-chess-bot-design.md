# Chess bot that plays like Justin: design

Status: draft for Justin's review. Nothing has been built.

Convention: **[Justin]** = decided by Justin. **[My call]** = chosen by the
writer, not yet agreed. **[Open]** = needs an answer.

## Intent

Justin wants a chess bot that plays the way *he* plays: his openings, his
typical mistakes, his strength. Anyone should be able to play it from a link
in a repo README. He has 10,000+ chess.com games to learn from and a Google
Colab Pro account; budget for anything else is effectively zero.

The target is a bot that predicts the move Justin would play, not the best
move. A bot that plays better than Justin has failed.

## Goals

1. Predict Justin's next move on games the bot has never seen. [Justin]
2. Match Justin's average strength and error profile (centipawn loss, blunder
   rate), not beat it. [Justin]
3. Follow Justin's real opening choices while in his repertoire. [Justin]
4. Be playable by anyone via a link, at near-zero running cost. [Justin]

## Non-goals

- Resigning, draw offers, or clock play. The bot never resigns. [Justin]
  Visitors get no time control; the clock is out of scope for v1. [My call]
- Visitor controls beyond choosing a color. No difficulty or randomness
  slider. [Justin]
- Playing as strongly as possible.
- The language-model side project ("bedtime-LM"). Separate design.

## Anti-goals (a technical success that still counts as failure)

- A bot stronger than Justin, or one that never blunders.
- Test-set leakage: any evaluation number computed on games the opening
  tree, position lookup, or fine-tune has seen.
- Raw PGNs (which contain opponents' usernames) ending up in a public repo.
- Licensing terms violated by the published repo or site.

## Data

- Source: chess.com games for Justin's account, via the public archive API.
  [Justin: chess.com, username Justin0921.]
- All time controls are kept for v1. [Justin] Every metric is also reported
  per time control, so we can see whether one bot fits all of them. [My call]
- Target strength is Justin's lifetime average across all games, not his
  current rating, because it uses all the data. [Justin] Ratings are read
  from the PGN headers (WhiteElo/BlackElo); no recency weighting.
- Split by date: oldest ~85-90% for training, newest ~10-15% held out for
  testing. The test set is touched only for final measurements. [My call]
- Filter out aborted games and games with too few moves to be meaningful.
  [My call]

## Model

- Base model: Maia 3 (University of Toronto CSSLab, "Chessformer", rating
  range 600-2600, PyTorch checkpoints at 23M and 79M parameters). [Justin]
- Moves are sampled from the model's probability distribution, never
  argmax, with one fixed temperature tuned once against the centipawn-loss
  metric. No search. [My call]
- Fine-tuning on Justin's training games is optional and happens only if it
  beats the un-fine-tuned harness on the held-out test set. Done in Colab
  Pro. [My call]

## Bot behavior, in order of priority

1. **Opening tree.** A table keyed by board position (FEN, so transpositions
   merge), built from Justin's *training* games only, giving how often he
   played each move. While the position is in the table, sample by those
   frequencies. [Justin: build it ourselves from the PGN instead of
   exporting from openingtree.com.] OpeningTree remains a visual
   cross-check.
2. **Position lookup.** The same table covers every position seen in the
   training games, not only the opening. A position is used only if it was
   seen at least 3 times, so a one-off move does not dominate. [Justin: 3]
3. **Maia 3 fallback** for everything else, using Justin's rating as the
   conditioning input. [My call on using the rating input as described.]
4. **Never resigns.** [Justin]

## Success measures [Justin]

1. **Next-move accuracy on the held-out games.** Top-1 and top-3, split by
   opening, middlegame, and endgame. Baseline to beat: plain Maia 3 at
   Justin's rating. [Top-1/top-3 and the baseline are My call.]
2. **Game quality by centipawn loss.** Stockfish evaluates Justin's games and
   the bot's games. The goal is for the bot's average centipawn loss and
   blunder rate to *match* Justin's, not to be lower. [Matching rather than
   minimizing is My call, from the "plays like me" goal.]
3. **LLM judge of playstyle**, later. Blind: shown games by Justin, the bot,
   and plain Maia, it tries to tell them apart. Treated as a secondary,
   noisy signal. [Justin wants it; timing "later" is Justin's.]

### Supporting metrics [My call, proposed 2026-11-02; Justin to confirm]

4. **Likelihood of Justin's real move.** Average log-probability (or
   perplexity) the bot assigns to the move Justin played. Top-1 accuracy
   understates a sampling bot, because humans are not deterministic.
5. **Error-profile match.** Share of moves losing more than 100/200/300
   centipawns, by phase and by time control, compared with Justin's.
6. **Opening match.** Divergence between the bot's move distribution and
   Justin's over the first ~10 plies, and how deep the bot stays in his
   book. Also lookup coverage: the share of held-out positions found in the
   table (count >= 3), and accuracy when found versus when not.
7. **Blind human test.** Friends (or Justin) shown games from Justin and the
   bot and asked which is which.
8. **Engineering checks.** 100% legal moves, under ~1 second per move in
   the browser, and a page load time that is acceptable on a phone.

## Build order [Justin, 2026-11-02]

Focus now: the data pipeline, the opening and position table, the Maia 3
fallback running locally, and the evaluation metrics. Hosting comes only
after the harness works the way we want, and the browser/ONNX route is one
option for it, not a requirement. Everything under Delivery is deferred.

## Delivery (deferred)

- A static site (GitHub Pages or Justin's own site) with a board UI where a
  visitor picks White or Black and plays. The README links to it. [Justin
  wants link-from-README play; static site is My call.]
- The model runs in the visitor's browser (ONNX export, onnxruntime-web), so
  hosting costs nothing and scales. Likely candidate is the 23M model.
  [My call. Feasibility and load time unverified.]
- The board UI must be usable on a phone (touch-friendly, no install). [My call]
- The site ships the processed lookup table (not raw PGNs), so Justin's
  repertoire and positions he has played are publicly readable. [Open:
  Justin's comfort with that.]
- Optional later: a Lichess bot account so people can challenge it from
  Lichess. Requires an always-on host and an account with no games played.
  Not part of v1. [My call]

## Constraints

- Budget: effectively $0. Compute is a laptop plus Colab Pro.
- Licensing: the Maia 3 repo is AGPL-3.0 and the 79M card says AGPLv3; the
  5M and 23M cards only say "see repo for code/weights license". Treat
  everything, including converted and fine-tuned weights and any copied
  code, as AGPL-3.0: the public repo and site must carry that license, the
  source, and attribution (cite the Chessformer paper). Not legal advice.
  See "Verification results" below.

## Left to the builder

- Project structure, languages, and libraries (python-chess, PyTorch,
  onnxruntime-web, a JS board library, and so on).
- Exact thresholds: minimum position count, sampling temperature, fine-tune
  learning rate and epochs, the blunder cutoff in centipawns.
- Which Maia 3 size to ship, based on measured accuracy against load time
  and speed on a phone.
- How rating drift is weighted in training.

## Verification results (checked 2026-11-02)

1. **License: checked.** Repo `CSSLab/maia3` is AGPL-3.0. The 79M card says
   AGPLv3; the 5M and 23M cards defer to the repo. Plan on AGPL-3.0 for all.
2. **ONNX export: deferred until the harness works [Justin]. Not run,
   read only.** The model is ordinary PyTorch
   (nn.MultiheadAttention with a float attention bias, einsum, LayerNorm,
   RMSNorm, GELU) over a fixed 64 square tokens, which looks exportable.
   Checkpoint sizes (fp32 .pt): 5M = 21 MB, 23M = 92 MB, 79M = 316 MB. The
   79M is too large for a casual web page; 23M is borderline and would need
   fp16 or int8 quantization; 5M is comfortable. A real export test was
   blocked by the environment's permissions and still needs doing.
3. **Fine-tuning code: not in the public repo.** The repo's README says it
   holds inference code only; the HF cards mention a training repo and
   `ablate_size.sh`, which I could not find. The model definition, board
   tokenization, move dictionary, and Elo conditioning are all present, so
   writing our own fine-tune loop (cross-entropy on the move logits with a
   legal-move mask) looks feasible. Unverified on Colab.
4. **chess.com data: checked.** `Justin0921` has 71 monthly archives
   (2020/12 to 2026/11; one month in that span has none) and 10,151 games
   total, with no errors or rate limiting at 0.3 s between requests.
   By time class: bullet 5,222, blitz 4,363, rapid 556, daily 10. By rules:
   standard chess 10,045, bughouse 65 (these lack a PGN), chess960 26,
   odds chess 10, king of the hill 5. Keep standard chess only (~10,045).

### Findings that change the design

- **Over half the games are bullet.** Bullet play is full of time-pressure
  moves a no-clock bot cannot reproduce. Report metrics per time control
  and decide later whether the bot represents all play or a subset.
- **Maia 3 is conditioned on Lichess-scale ratings**, with separate
  "self Elo" and "opponent Elo" inputs. chess.com ratings differ from
  Lichess ratings, so the Elo fed to the model should be calibrated:
  pick the value that maximizes move-match on Justin's training games.
  [My call]
- Some Maia 3 checkpoints may also take time-control inputs (base,
  increment, clock). Whether the released ones do is unchecked. [Open]
- Stockfish (for centipawn loss) is not installed; it is free to install.
  torch 2.11 and onnxruntime 1.21 are already installed locally.
