"""One listening file for choosing spellings by ear: per word, a beep, then each spelling variant in order.

    python tools/ab_listen.py out/ab-ya --seed 0
Prints the numbered order, so the owner can answer «noyabr: 3, sentabr: 1 …».
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rows = [r for r in json.loads((args.folder / "ab.json").read_text(encoding="utf-8")) if r["seed"] == args.seed]
sr = 24000
beep = 0.25 * np.sin(2 * np.pi * 880 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
beep *= np.minimum(1, np.minimum(np.arange(len(beep)), np.arange(len(beep))[::-1]) / 200)
parts, words = [], []
for r in rows:
    if r["word"] not in words:
        words.append(r["word"])
        parts += [np.zeros(int(0.8 * sr), np.float32), beep, np.zeros(int(0.5 * sr), np.float32)]
        print(f"\n{r['word']}:")
    y, file_sr = sf.read(args.folder / r["file"], dtype="float32")
    assert file_sr == sr
    parts += [y, np.zeros(int(0.7 * sr), np.float32)]
    n = sum(1 for x in rows[:rows.index(r) + 1] if x["word"] == r["word"])
    print(f"  {n}) {r['spelling']:10} «{r['sentence']}»")
sf.write(args.folder / "_tanlov.wav", np.concatenate(parts), sr)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4", "-i", str(args.folder / "_tanlov.wav"),
                "-codec:a", "libmp3lame", "-b:a", "160k", str(args.folder / "tanlov.mp3")], check=True)
(args.folder / "_tanlov.wav").unlink()
print("\nsaved", args.folder / "tanlov.mp3")
