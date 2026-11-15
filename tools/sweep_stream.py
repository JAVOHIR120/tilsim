"""Try streaming settings with one model load: dtype, S3Gen reference length, flow steps, chunk sizes.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/sweep_stream.py --out out/sweep/laptop.json

Each setting runs the short and medium bench texts (tools/bench_stream.py) twice and keeps the median.
Settings are applied in order, cheapest first. fp32 runs must come before bf16/fp16 ones (the cast is one-way).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT), str(ROOT / "tools")]

DEFAULT = [
    {"name": "fp32 base", "dtype": "fp32", "dec": 10, "cfm": 10, "first": 10, "chunk": 30},
    {"name": "fp32 dec4", "dtype": "fp32", "dec": 4, "cfm": 10, "first": 10, "chunk": 30},
    {"name": "fp32 dec4 cfm6", "dtype": "fp32", "dec": 4, "cfm": 6, "first": 10, "chunk": 30},
    {"name": "fp32 dec4 cfm6 c50", "dtype": "fp32", "dec": 4, "cfm": 6, "first": 10, "chunk": 50},
    {"name": "half dec4 cfm6 c50", "dtype": "half", "dec": 4, "cfm": 6, "first": 10, "chunk": 50},
    {"name": "half dec4 cfm6 f8 c50", "dtype": "half", "dec": 4, "cfm": 6, "first": 8, "chunk": 50},
    {"name": "half dec4 cfm4 c50", "dtype": "half", "dec": 4, "cfm": 4, "first": 10, "chunk": 50},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "sweep" / "sweep.json")
    ap.add_argument("--texts", default="short,medium")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--configs", help="JSON list replacing the default settings")
    ap.add_argument("--graphs", action="store_true", help="load T3 in half precision with CUDA graphs (engine/fastt3.py)")
    ap.add_argument("--fast-flow", action="store_true", help="flow UNet in half precision with CUDA graphs (engine/fastflow.py)")
    args = ap.parse_args()

    import torch

    import bench_stream as bench
    import stream

    configs = json.loads(args.configs) if args.configs else DEFAULT
    half = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    s = stream.Streamer.load("ovozim", dtype=half if args.graphs else "fp32", graphs=args.graphs,
                             flow_half=args.fast_flow, flow_graphs=args.fast_flow, dec_cond=4 if args.fast_flow else None)
    ref = s.engine  # for re-preparing the S3Gen reference at another length
    import os
    profile = json.loads((ROOT / "data" / "voices" / "ovozim.json").read_text(encoding="utf-8"))
    ref_path = os.environ.get("TILSIM_REF") or profile["ref"]
    t3t, s3t = bench.Timer(torch), bench.Timer(torch)
    s.t3_timer, s.s3_timer = t3t, s3t
    current = {"dtype": half if args.graphs else "fp32", "dec": 4 if args.fast_flow else 10}
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        dtype = half if cfg["dtype"] == "half" else cfg["dtype"]
        if dtype != current["dtype"]:
            s.engine.t3.to(stream.DTYPES[dtype])
            s.engine.conds.t3.to(dtype=stream.DTYPES[dtype])
            s.dtype_name, s.dtype = dtype, stream.DTYPES[dtype]
            current["dtype"] = dtype
        if cfg["dec"] != current["dec"]:
            ref.DEC_COND_LEN = int(cfg["dec"] * stream.SR)
            import librosa
            wav, _ = librosa.load(ref_path, sr=stream.SR)
            ref.conds.gen = ref.s3gen.embed_ref(wav[:ref.DEC_COND_LEN], stream.SR, device=ref.device)
            current["dec"] = cfg["dec"]
        s.cfm_steps, s.first_chunk, s.chunk = cfg["cfm"], cfg["first"], cfg["chunk"]
        s.warmup()
        row = {**cfg, "dtype": dtype, "device": dev}
        for name in args.texts.split(","):
            runs = []
            for r in range(args.runs):
                t3t.total = s3t.total = 0.0
                torch.cuda.reset_peak_memory_stats()
                res = bench.run_stream(s, bench.TEXTS[name], t3t, s3t)
                res["rtf"] = res["gen"] / max(res["audio"], 1e-6)
                res["t3"], res["s3"] = t3t.total, s3t.total
                res["vram"] = torch.cuda.max_memory_reserved() / 2**30
                runs.append(res)
            row[name] = {k: round(statistics.median([x[k] for x in runs if x.get(k) is not None]), 3)
                         for k in ("ttfa", "rtf", "audio", "t3", "s3", "vram", "min_buffer") if runs[0].get(k) is not None}
        results.append(row)
        print(f"{cfg['name']:<24} " + "  ".join(
            f"{n}: ttfa {row[n]['ttfa']:.2f} rtf {row[n]['rtf']:.2f} (t3 {row[n]['t3']:.1f} s3 {row[n]['s3']:.1f}) "
            f"vram {row[n]['vram']:.2f}" for n in args.texts.split(",")), flush=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("JSON " + json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
