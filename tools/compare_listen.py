"""Side-by-side listening file: the same sentences in several voices, a beep before each sentence.

    python tools/compare_listen.py --ids t02,t07,t12 --voices "Sardor=out/bench-azure-sardor" "Madina=out/bench-azure-madina" "Biz=out/exp-K-judge" --out out/solishtirish.mp3
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True)
ap.add_argument("--voices", nargs="+", required=True, help="Name=folder (folder has manifest.json)")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
sr = 24000
beep = 0.25 * np.sin(2 * np.pi * 880 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
voices = []
for spec in args.voices:
    name, folder = spec.split("=", 1)
    rows = {r["id"]: r for r in json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))["lines"]}
    voices.append((name, Path(folder), rows))
parts = []
for n, item_id in enumerate(args.ids.split(","), 1):
    text = next(v[2][item_id]["text"] for v in voices if item_id in v[2])
    print(f"{n}. «{text}»: " + ", ".join(f"{k}) {name}" for k, (name, _, _) in enumerate(voices, 1)))
    parts += [np.zeros(int(0.6 * sr), np.float32), beep, np.zeros(int(0.4 * sr), np.float32)]
    for name, folder, rows in voices:
        y, file_sr = sf.read(folder / rows[item_id]["file"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        y = y / max(1e-6, np.abs(y).max()) * 0.7  # same peak level for a fair listen
        parts += [y, np.zeros(int(0.8 * sr), np.float32)]
tmp = args.out.with_suffix(".wav")
sf.write(tmp, np.concatenate(parts), sr)
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4", "-i", str(tmp), "-codec:a", "libmp3lame",
                "-b:a", "160k", str(args.out)], check=True)
tmp.unlink()
print("saved", args.out)
