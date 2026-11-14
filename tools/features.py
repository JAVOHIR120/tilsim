"""Audio features for "listening with eyes": mel spectrogram, pitch (F0) and loudness per file -> one .npz each.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/features.py out/first-uz/*.wav --out out/first-uz/_feat
Run tools/plot_features.py (any Python with matplotlib) on the folder afterwards.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--manifest", type=Path, help="manifest.json from engine/uz_tts.py, for the text of each file")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
texts = {}
if args.manifest and args.manifest.exists():
    texts = {r["file"]: r["spoken"] for r in json.loads(args.manifest.read_text(encoding="utf-8"))["lines"]}

for path in args.files:
    y, sr = librosa.load(path, sr=24000)
    hop = 240  # 10 ms
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=100, fmax=12000), ref=np.max)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    np.savez_compressed(args.out / f"{path.stem}.npz", mel=mel.astype(np.float32), f0=f0.astype(np.float32),
                        voiced=voiced, rms_db=(20 * np.log10(rms + 1e-9)).astype(np.float32), sr=sr, hop=hop,
                        text=texts.get(path.name, path.name))
    print("ok", path.name)
