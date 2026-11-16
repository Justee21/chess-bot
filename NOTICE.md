# Attribution

This project fine-tunes the `maia3-23m` Maia 3 model and uses its inference implementation.

- Authors / source: University of Toronto CSSLab, [Maia 3](https://github.com/CSSLab/maia3).
- Exact code revision: `1e13597c42d4858b7cfd7cfdae01e297263364b2`.
- Original base weights: [UofTCSSLab/Maia3-23M](https://huggingface.co/UofTCSSLab/Maia3-23M).
- Original license: [AGPL-3.0](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/LICENSE).
- Paper: Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson, *Chessformer: A Unified Architecture for Chess Modeling* (2026). [Paper](https://arxiv.org/abs/2605.19091).

The released `finetuned-23m-clean.pt` modifies Maia 3 through supervised fine-tuning on Justin Roberts's own moves. Its SHA-256 and deployment parameters are recorded in `config/final.json`. The position table is built from his pre-test games.

Lichess integration uses the separate [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) project at revision `06f4f10b5a5a70549b9fc5f8f759a532723b5dab`, licensed AGPL-3.0-or-later. Its code is not vendored in this repository.
