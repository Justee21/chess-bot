# Play Justin in a browser

The browser version runs the frozen fine-tune 2 policy locally in a Web Worker. It needs static HTTPS hosting, not a model server or Lichess token. Click **Play Justin** to load the model and reveal the board. Visitors choose White or Black and can start a new game.

The standalone demo is published at https://justee21.github.io/chess-bot/web/. The personal website embeds this same demo, so the model files are maintained in one repository. Files load only after the visitor clicks Play Justin.

## What ships

- Full-precision FP32 ONNX export of `finetuned-23m-clean.pt`, with both Elo inputs fixed at $1500$.
- Three binary model chunks, each below GitHub's per-file size limit. The worker reassembles them and verifies their SHA-256 hashes, plus the complete model hash.
- A compact $214\,\mathrm{KB}$ table containing all $2{,}580$ positions with total count at least $3$. Positions below that threshold never fire in the original policy, so excluding them preserves lookup behavior. `manifest.json` retains the source table and checkpoint hashes.
- The same temperature $1.2$, relative cutoff $0.05$, table-first fallback, history encoding, move vocabulary, promotion mapping, and immediate-draw avoidance as the Python policy.
- chess.js $1.4.0$ for rules and board history; ONNX Runtime Web $1.31.1$ using single-threaded WebAssembly.

No quantization, retraining, search, network inference, or model-policy adjustment is used. The original `config/final.json`, checkpoint, and table are unchanged. As with any floating-point runtime conversion, predictions are numerically close rather than bit-identical.

The ONNX model is about $91.6\,\mathrm{MB}$. Model, table, and browser runtime total roughly $106\,\mathrm{MB}$ on first use, before transfer compression. Model chunks and table are cached when the browser permits it. Network speed, memory, and browser/device performance affect loading; private browsing or cache eviction can require another download.

## Verification

On 2026-11-18:

- PyTorch versus native ONNX: identical top-1 moves at all $128$ synthetic test positions; maximum legal probability error below $0.000005$.
- Browser versus Python: exact input tokenization and matching policy source/playable move sets at $48$ test positions, including Black-to-move, promotion, castling, en passant, repetition, stalemate avoidance, and the fifty-move boundary. Maximum model probability error below $0.000006$.
- Browser table keys checked against all $2{,}580$ included positions.
- Browser evaluation median approximately $100\,\mathrm{ms}$ on the development Mac in Chromium with single-threaded WASM. This includes policy checks and is not a mobile-device benchmark.
- The WebKit test-engine download failed, so Safari/iPhone compatibility is not yet verified.
- Real UI checks cover White play and a model reply, Black orientation/opening move, new game, and a narrow viewport without horizontal overflow. A narrow Chromium viewport is not a real iPhone/Safari test.

See `docs/results/onnx-export-2026-11-18.json` and `docs/results/browser-validation-2026-11-18.json`. These are conversion/integration checks, not new strength or held-out accuracy experiments.

## Rebuild

With the existing Python environment activated and the released artifacts downloaded:

```bash
python -m pip install -e '.[browser]'
python scripts/export_browser.py --config config/final.json --out web --positions 128
npm ci
npm run vendor:web
npx playwright install chromium
npm run test:browser
```

The exporter updates `tests/browser-fixtures.json` and writes `docs/results/onnx-export.json`. Python package versions and torch version are recorded in the report. NPM dependencies are pinned in `package-lock.json`.

To preview without rebuilding:

```bash
python -m http.server 8000
```

Open http://localhost:8000/web/. Localhost is treated as a secure context; production must use HTTPS for hashing and caching APIs. The model and runtime assets are all same-origin, avoiding third-party runtime downloads or special cross-origin isolation headers.

The browser source and model use the repository's AGPL license. Vendored chess.js is BSD-2-Clause; ONNX Runtime Web is MIT. Their notices are in `web/vendor/`.
