# Second fine-tune: ignoring time-scramble moves (Colab runbook)

This trains a second model on the cleaned examples (bullet moves played with 5 seconds or less left that lost 200+ centipawns are removed). It writes a NEW file, `finetuned-23m-clean.pt`. Your first model, `finetuned-23m.pt`, is never touched.

Upload file on your Mac: `~/claude-projects/chess-bot/finetune-upload-clean.zip` (already built).

## Before you start (mistakes to avoid)

- Keep your laptop awake (plugged in, sleep off) and keep the Colab tab open the whole time.
- Colab wipes its disk if the runtime disconnects. The moment training finishes, download the checkpoint. Do not wait.

## Cells

1. Open https://colab.research.google.com and create a new notebook.
2. **Set the GPU first.** Runtime > Change runtime type > pick a GPU (T4 is fine) > Save. Do this before anything else.
3. Open the Files panel (folder icon, left sidebar). Click the `content` folder so it is open. If you do not see a `content` folder, click the folder-up arrow once. The upload must land INSIDE `content`, not in `/`. (Last time it landed in `/` and unzip could not find it.)
4. Drag `finetune-upload-clean.zip` onto the `content` folder, or use the upload button while `content` is open. Wait for the progress circle to finish. Then run this cell and check the zip is listed:

```python
!ls -la /content
```

5. Set up (run as one cell):

```python
%cd /content
!unzip -q finetune-upload-clean.zip -d chess-bot
%cd chess-bot
!pip install -q python-chess numpy
!pip install -q "git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2"
!pip install -q -e .
```

6. Train:

```python
!python -m chessbot finetune --examples data/finetune-clean --model maia3-23m --device cuda \
    --lr 1e-5 --epochs 3 --patience 1 --out finetuned-23m-clean.pt
```

7. Download it right away. Check the file exists and is about 108 MB:

```python
!ls -la finetuned-23m-clean.pt
```

Then in the Files panel open `chess-bot`, right-click `finetuned-23m-clean.pt` > Download.

8. On your Mac, put the file at `~/claude-projects/chess-bot/data/finetuned-23m-clean.pt`. Do not overwrite `data/finetuned-23m.pt`. Then tell Claude it is there.

## What the output should look like

- It prints validation loss and top-1 for epoch 0 (the untouched base model) and each later epoch.
- Epoch 0 shows `train_loss nan`. That is normal.
- The last line says which epoch was best and that it wrote `finetuned-23m-clean.pt`.
- `best epoch 0` means it never beat the base model. See the tuning notes in `docs/finetune-colab.md`.
- Out of GPU memory: add `--batch-size 32`.
