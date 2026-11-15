"""Render a test set through the streaming engine, so tools/eval_uz.py can score it (CER, UTMOS, similarity).

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/render_stream.py data/eval/test_uz.tsv out/cer/stream-cfm6 \
        --cfm-steps 6 --fast
    E:/tools/voiceclone/.venv/Scripts/python.exe tools/eval_uz.py out/cer/stream-cfm6

Writes one WAV per line (the chunks joined exactly as a player would play them) and manifest.json in the same
format as engine/uz_tts.py. Also reports the largest sample-to-sample jump inside each file and right at chunk
borders: a click would show as a border jump far above the file's own typical jumps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cfm-steps", type=int, default=10)
    ap.add_argument("--first-chunk", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--dec-cond", type=float, default=4.0)
    ap.add_argument("--fast", action="store_true", help="half-precision T3 + flow with CUDA graphs")
    ap.add_argument("--t3-dtype", default="", help="with --fast: keep the token model in this dtype (e.g. fp32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--english-ref", action="store_true",
                    help="reference text for the ASR check = English parts respelled (what an Uzbek listener should "
                         "hear); use for data/eval/test_mixed.tsv. TILSIM_ENGLISH=raw renders without respelling.")
    args = ap.parse_args()

    import soundfile as sf
    import torch

    import stream
    half = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    s = stream.Streamer.load("ovozim", dtype=(args.t3_dtype or half) if args.fast else "fp32", cfm_steps=args.cfm_steps,
                             first_chunk=args.first_chunk, chunk=args.chunk, dec_cond=args.dec_cond,
                             graphs=args.fast, flow_half=args.fast, flow_graphs=args.fast)
    s.warmup()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, border_ratio = [], []
    for ln in args.tsv.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        item_id, category, text = ln.split("\t", 2)
        pieces, borders, n = [], [], 0
        for ev in s.stream(text, seed=args.seed):
            if ev["type"] == "audio" and len(ev["pcm"]):
                if n:
                    borders.append(n)
                pieces.append(ev["pcm"])
                n += len(ev["pcm"])
        audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
        d = np.abs(np.diff(audio))
        typical = float(np.quantile(d, 0.999)) if len(d) else 0.0
        at_border = max((float(np.abs(audio[b] - audio[b - 1])) for b in borders), default=0.0)
        border_ratio.append(at_border / max(typical, 1e-6))
        sf.write(args.out / f"{item_id}.wav", audio, s.sr)
        from uznorm import normalize
        from uznorm.english import respell_english
        spoken = respell_english(normalize(text)) if args.english_ref else normalize(text)
        # eval_uz.py scores against "text": with --english-ref that is the respelled form, the original is kept
        rows.append({"file": f"{item_id}.wav", "id": item_id, "category": category, "seed": args.seed,
                     "text": spoken if args.english_ref else text, "original_text": text, "spoken": spoken, "seconds": round(len(audio) / s.sr, 2), "chunks": len(pieces),
                     "border_jump": round(at_border, 4), "typical_jump_p999": round(typical, 4)})
        print(f"{item_id} {rows[-1]['seconds']:5.1f} s, {len(pieces)} chunks, border/typical {border_ratio[-1]:.2f}",
              flush=True)
    (args.out / "manifest.json").write_text(json.dumps(
        {"engine": "stream", "cfm_steps": args.cfm_steps, "first_chunk": args.first_chunk, "chunk": args.chunk,
         "dec_cond": args.dec_cond, "fast": args.fast, "lines": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"worst chunk-border jump / file's 99.9th-percentile jump: {max(border_ratio):.2f} "
          f"(below ~1 means no border stands out)")


if __name__ == "__main__":
    main()
