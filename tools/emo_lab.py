"""Emotion lab: many takes of the founder's voice in a mood, for building and testing the emotion bank.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/emo_lab.py --tsv data/eval/emo_seed.tsv --plan plan.json --out-dir out/emo/A
    ... --tsv data/eval/emo_neutral.tsv --bank data/voices/ovozim.json --emotions iliq,quvnoq --intensity 0.6 --out-dir out/emo/B

Two ways to say what to make:
- --plan: a JSON list of {"ids": "quvnoq_" (id prefix) | "*", "grid": [[exaggeration, cfg_weight, temperature], ...],
  "prefix": {"wav", "text", "end"} (optional), "prompt_wav" (optional), "tag": "name"}. Every matching sentence is made
  with every grid point and every seed.
- --bank + --emotions + --intensity: the engine's own style() for each emotion (what the API will do), every sentence
  in every emotion. That is the test of the finished bank.
Uses the streaming engine's whole-sentence path (T3 in bf16 with CUDA graphs, flow with --cfm-steps), so takes are
made the way the server makes them. Writes OUT/manifest.json for tools/eval_uz.py and tools/gemini_judge.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))
import stream  # noqa: E402
import uz_tts  # noqa: E402
from uznorm import normalize  # noqa: E402


def read_tsv(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            i, cat, text = ln.split("\t", 2)
            out.append({"id": i, "category": cat, "text": text})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--profile", default="ovozim", help="voice profile to load (data/voices/<name>.json)")
    ap.add_argument("--emotions", default="", help="with the profile's emotion bank: comma list of emotions to make")
    ap.add_argument("--intensity", default="0.6", help="comma list of strengths for --emotions")
    ap.add_argument("--match", action="store_true", help="with --emotions: only sentences whose category is that emotion")
    ap.add_argument("--lang", default="uz")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cfm-steps", type=int, default=10)
    ap.add_argument("--dec-cond", type=float, default=10)
    args = ap.parse_args()
    items = read_tsv(args.tsv)
    seeds = [int(s) for s in args.seeds.split(",")]

    import torchaudio
    t0 = time.time()
    s = stream.Streamer.load(args.profile, dtype="bf16", cfm_steps=args.cfm_steps, dec_cond=args.dec_cond, graphs=True,
                             streams=1, flow_half=True, flow_graphs=True)
    print(f"ready in {time.time() - t0:.0f} s, VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    eng = s.engine
    jobs = []   # (item, tag, style, params-for-manifest)
    if args.plan:
        for k, entry in enumerate(json.loads(args.plan.read_text(encoding="utf-8"))):
            base = s.style(args.lang)
            prefix = base.prefix
            if entry.get("prefix"):
                pf = entry["prefix"]
                prefix = uz_tts.Prefix(eng, ROOT / pf["wav"], pf["end"], pf["text"])
            prompt = stream.prompt_tokens(eng, ROOT / entry["prompt_wav"]) if entry.get("prompt_wav") else \
                base.cond.cond_prompt_speech_tokens
            for g, (ex, cfg, temp) in enumerate(entry["grid"]):
                params = {**base.params, "exaggeration": ex, "cfg_weight": cfg, "temperature": temp}
                st = dataclasses.replace(base, prefix=prefix, params=params, cond=s._cond(base.cond, prompt, ex))
                tag = f"{entry.get('tag', f'p{k}')}g{g}"
                for it in items:
                    if entry["ids"] == "*" or it["id"].startswith(entry["ids"]):
                        jobs.append((it, tag, st, {"exaggeration": ex, "cfg_weight": cfg, "temperature": temp,
                                                   "prefix": entry.get("prefix", {}).get("wav"),
                                                   "prompt_wav": entry.get("prompt_wav")}))
    else:
        for emo in [e for e in args.emotions.split(",") if e]:
            for x in [float(v) for v in args.intensity.split(",")]:
                st = s.style(args.lang, emo, x)
                tag = f"{emo}-x{int(round(x * 100)):03d}"
                for it in items:
                    if args.match and it["category"] != emo:
                        continue
                    jobs.append((it, tag, st, {**{k: round(v, 3) for k, v in st.params.items()}, "emotion": emo,
                                               "intensity": x}))
    print(f"{len(jobs) * len(seeds)} takes to make", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mpath = args.out_dir / "manifest.json"
    rows = json.loads(mpath.read_text(encoding="utf-8"))["lines"] if mpath.exists() else []
    done = {r["file"] for r in rows if (args.out_dir / r["file"]).exists()}
    rows = [r for r in rows if r["file"] in done]
    for it, tag, st, meta in jobs:
        spoken = normalize(it["text"])
        for seed in seeds:
            name = f"{it['id']}__{tag}__s{seed}.wav"
            if name in done:
                continue
            t1 = time.time()
            torch.manual_seed(seed)
            audio = np.concatenate(list(s.sentence(spoken, seed=seed, whole=True, style=st)) or [np.zeros(0, np.float32)])
            took = time.time() - t1
            torchaudio.save(str(args.out_dir / name), torch.from_numpy(audio).unsqueeze(0), s.sr)
            rows.append({"file": name, "id": it["id"], "category": it["category"], "tag": tag, "seed": seed,
                         "text": it["text"], "spoken": spoken, "seconds": round(len(audio) / s.sr, 2),
                         "gen_seconds": round(took, 2), **meta})
            mpath.write_text(json.dumps({"lines": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name}  {len(audio) / s.sr:.2f} s audio in {took:.2f} s", flush=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; {len(rows)} takes in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
