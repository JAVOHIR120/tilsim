"""Transcribe a few files with the Uzbek Whisper (optionally only the first N seconds).

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/asr_quick.py a.wav b.wav --head 1.5
"""
import argparse
import sys
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_uz import ASR_DIR  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+", type=Path)
ap.add_argument("--head", type=float, default=0, help="also transcribe only the first N seconds")
args = ap.parse_args()
proc = WhisperProcessor.from_pretrained(ASR_DIR)
model = WhisperForConditionalGeneration.from_pretrained(ASR_DIR, dtype=torch.float16).to("cuda").eval()  # no accelerate here


def hear(audio):
    feats = proc([audio], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.inference_mode():
        ids = model.generate(feats, language="uz", task="transcribe", num_beams=5, max_new_tokens=200)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


for f in args.files:
    y, _ = librosa.load(f, sr=16000)
    print(f"{f.name}  ({len(y) / 16000:.2f} s): {hear(y)}", flush=True)
    if args.head:
        print(f"{'':>8}first {args.head} s: {hear(y[:int(args.head * 16000)])}", flush=True)
