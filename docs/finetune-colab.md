# Fine-tuning Maia on your games (Colab Pro runbook)

Training the 23M model needs a GPU. Everything else runs on your Mac. Nothing here overwrites the base Maia weights: the result is a new file.

## 1. On your Mac: export the examples

```bash
cd ~/claude-projects/chess-bot
.venv/bin/python -m chessbot export-examples
zip -r finetune-upload.zip src pyproject.toml data/finetune
```

`export-examples` writes `data/finetune/train.npz` and `val.npz` (about 10 MB together). It prints the game counts (train / validation / test); the test games are counted but never written, so Colab never sees them.

## 2. In Colab: set up

1. Open https://colab.research.google.com, create a new notebook, and choose **Runtime > Change runtime type > a GPU** (T4 is fine).
2. Upload `finetune-upload.zip` (left sidebar, Files) and run:

```python
!unzip -q finetune-upload.zip -d chess-bot
%cd chess-bot
!pip install -q python-chess numpy
!pip install -q "git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2"
!pip install -q -e .
```

(Colab already has `torch`. The commit above is the exact maia3 version the local code was tested against.)

## 3. In Colab: train

Starting values:

```python
!python -m chessbot finetune --examples data/finetune --model maia3-23m --device cuda \
    --lr 1e-5 --epochs 3 --patience 1 --out finetuned-23m.pt
```

It prints validation loss and top-1 for epoch 0 (the untouched base model) and each epoch after, then saves the best epoch. Reading the output:

- `best epoch 0` means the fine-tune never beat the base model on validation. Try a smaller `--lr 3e-6`, or the options below.
- Validation loss going up after epoch 1 while train loss keeps falling means it is memorizing your games. Re-run with a **new** `--out` name (it refuses to overwrite) and one of:
  - `--kl-weight 0.1` keeps its moves close to base Maia's.
  - `--freeze-layers 4` leaves the first four transformer layers alone.
  - A lower `--lr`.
- If Colab runs out of GPU memory, add `--batch-size 32`.

Try one change at a time and keep the run with the best validation top-1.

## 4. Get the file back

Download `finetuned-23m.pt` from the Colab Files sidebar (about 92 MB) and put it in `~/claude-projects/chess-bot/data/finetuned-23m.pt`. Do not put it in the Hugging Face cache folder. Then tell Claude it's there and the comparison against base Maia runs from it.

## Quick local check (optional)

A tiny run on the Mac proves the pipeline works before you spend Colab time:

```bash
.venv/bin/python -m chessbot finetune --model maia3-5m --max-examples 2000 --epochs 1 --out data/smoke-5m.pt
```

Delete `data/smoke-5m.pt` afterwards.
