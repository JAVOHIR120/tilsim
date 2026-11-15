"""Latency benchmark for Tilsim speech: time to first audio, real-time factor, VRAM.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/bench_stream.py --mode baseline     # today's whole-text path
    E:/tools/voiceclone/.venv/Scripts/python.exe tools/bench_stream.py --mode stream        # engine/stream.py
    ... --mode stream --dtype bf16 --cfm-steps 6 --json out/bench/laptop-stream.json

On Modal (T4, warm): E:/tools/modal-venv/Scripts/modal.exe run deploy/modal_app.py::bench --mode stream

Numbers per text (median over --runs after one warm-up):
  ttfa  seconds from the call until the first audio sample is ready
        (baseline: the whole text, because /v1/tts answers only when everything is done)
  rtf   generation seconds / audio seconds (below 1 = faster than real time)
  vram  peak torch allocation and reserved memory during the run, GB
  t3/s3 seconds spent in the token model (T3) and in the vocoder (S3Gen: flow + HiFiGAN)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT)]

TEXTS = {
    # 9 words
    "short": "Salom! Bugun siz bilan qisqa gapirish mashg‘ulotini birga o‘tkazamiz.",
    # 30 words
    "medium": ("Birinchi qismda o‘zingiz haqingizda bir nechta savol beraman. Iltimos, javob berishda shoshilmang, "
               "fikringizni to‘liq va aniq ayting. Agar savolni tushunmasangiz, qayta so‘rashingiz mumkin. "
               "Qani, boshladik, siz hozir qaysi shaharda yashaysiz?"),
    # 80 words
    "long": ("Endi ikkinchi qismga o‘tamiz. Sizga bitta mavzu beraman va bir daqiqa tayyorlanish uchun vaqt bo‘ladi. "
             "Shundan keyin shu mavzu bo‘yicha bir yoki ikki daqiqa gapirasiz. Tayyorlanish paytida qog‘ozga qisqa "
             "eslatmalar yozib olishingiz mumkin. Mavzu shunday: o‘zingiz yaxshi ko‘radigan joy haqida gapirib bering. "
             "U qayerda joylashgan, u yerga qachon va kim bilan borgansiz, u yerda nimalar qilasiz va nega aynan shu joy "
             "sizga yoqishini tushuntiring. Vaqt tugaganda men sizni to‘xtataman. Tayyor bo‘lsangiz, boshlashingiz mumkin. "
             "Sizga omad tilayman, vaqt aynan hozirdan boshlanadi."),
}


def word_count(text: str) -> int:
    return len(text.split())


class Timer:
    """Wraps a function and adds up the wall time spent inside it (synchronising CUDA, so GPU work is counted)."""

    def __init__(self, torch):
        self.torch, self.total = torch, 0.0

    def wrap(self, fn):
        def inner(*a, **kw):
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            self.total += time.perf_counter() - t0
            return out
        return inner


def run_baseline(voice, text: str, t3t: Timer, s3t: Timer) -> dict:
    from uznorm import normalize, split_sentences
    sentences = split_sentences(normalize(text), max_chars=200)
    t0 = time.perf_counter()
    first = []
    audio = voice.synth(sentences, 0, progress=lambda done: first.append(time.perf_counter()) if done == 1 else None)
    total = time.perf_counter() - t0
    secs = len(audio) / voice.sr
    return {"ttfa": total, "first_sentence": first[0] - t0 if first else total, "gen": total, "audio": secs}


def run_stream(streamer, text: str, t3t: Timer, s3t: Timer) -> dict:
    t0 = time.perf_counter()
    ttfa, samples, chunks, gaps, last = None, 0, 0, [], None
    for item in streamer.stream(text):
        if item["type"] != "audio":
            continue
        now = time.perf_counter()
        if ttfa is None:
            ttfa = now - t0
        else:
            # how far ahead of playback we stay: audio already produced minus time spent
            gaps.append(samples / streamer.sr - (now - t0 - ttfa))
        samples += len(item["pcm"])
        chunks += 1
        last = now
    total = (last or time.perf_counter()) - t0
    secs = samples / streamer.sr
    return {"ttfa": ttfa or total, "gen": total, "audio": secs, "chunks": chunks,
            "min_buffer": round(min(gaps), 3) if gaps else None}


def run_parallel(streamer, text: str, n: int) -> dict:
    """n listeners at once (each its own thread and slot): worst TTFA, worst RTF, worst buffer margin."""
    import threading
    out = [None] * n

    def one(i):
        out[i] = run_stream(streamer, text, None, None)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in out:
        r["rtf"] = r["gen"] / max(r["audio"], 1e-6)
    return {"streams": n, "ttfa_max": round(max(r["ttfa"] for r in out), 3),
            "rtf_max": round(max(r["rtf"] for r in out), 3),
            "min_buffer_min": min((r["min_buffer"] for r in out if r["min_buffer"] is not None), default=None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", default="", help="stream mode: also run N listeners at once, e.g. 2,3")
    ap.add_argument("--mode", choices=["baseline", "stream"], default="baseline")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--texts", default="short,medium,long")
    ap.add_argument("--dtype", default="fp32", help="stream mode: fp32 | fp16 | bf16 for the token model")
    ap.add_argument("--cfm-steps", type=int, default=10, help="stream mode: flow-matching steps in the vocoder")
    ap.add_argument("--first-chunk", type=int, default=0, help="stream mode: tokens in the first chunk (0 = default)")
    ap.add_argument("--chunk", type=int, default=0, help="stream mode: tokens per later chunk (0 = default)")
    ap.add_argument("--dec-cond", type=float, default=0, help="stream mode: seconds of reference for S3Gen (0 = 10)")
    ap.add_argument("--graphs", action="store_true", help="stream mode: CUDA-graph T3 (needs --dtype bf16/fp16)")
    ap.add_argument("--streams", type=int, default=1, help="stream mode: parallel slots (each its own KV cache)")
    ap.add_argument("--fast-flow", action="store_true", help="stream mode: flow UNet in half precision + CUDA graphs")
    ap.add_argument("--json", type=Path, help="also write the results here")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import torch
    import serve
    import uz_tts

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    t3t, s3t = Timer(torch), Timer(torch)
    t_load = time.perf_counter()
    if args.mode == "baseline":
        voice = serve.Voice("ovozim")
        uz_tts._continue_tokens = t3t.wrap(uz_tts._continue_tokens)
        voice.engine.s3gen.inference = s3t.wrap(voice.engine.s3gen.inference)
        runner, target = run_baseline, voice
        voice.synth(["Salom."])  # warm-up
    else:
        import stream
        streamer = stream.Streamer.load("ovozim", dtype=args.dtype, cfm_steps=args.cfm_steps,
                                        first_chunk=args.first_chunk or None, chunk=args.chunk or None,
                                        dec_cond=args.dec_cond or None, graphs=args.graphs, streams=args.streams,
                                        flow_half=args.fast_flow, flow_graphs=args.fast_flow)
        streamer.t3_timer, streamer.s3_timer = t3t, s3t
        runner, target = run_stream, streamer
        streamer.warmup()
    load_s = time.perf_counter() - t_load
    print(f"{dev}: loaded + warmed in {load_s:.0f} s ({args.mode}, {args.dtype}, cfm {args.cfm_steps})", flush=True)

    rows = []
    for name in args.texts.split(","):
        text = TEXTS[name]
        results = []
        for r in range(args.runs):
            t3t.total = s3t.total = 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            res = runner(target, text, t3t, s3t)
            res["t3"], res["s3"] = t3t.total, s3t.total
            if torch.cuda.is_available():
                res["vram_alloc"] = torch.cuda.max_memory_allocated() / 2**30
                res["vram_reserved"] = torch.cuda.max_memory_reserved() / 2**30
            res["rtf"] = res["gen"] / max(res["audio"], 1e-6)
            results.append(res)
            print(f"  {name} run {r + 1}: ttfa {res['ttfa']:.2f} s, rtf {res['rtf']:.2f}, audio {res['audio']:.1f} s",
                  flush=True)
        med = {k: round(statistics.median([x[k] for x in results if x.get(k) is not None]), 3)
               for k in results[0] if isinstance(results[0][k], (int, float))}
        rows.append({"text": name, "words": word_count(text), **med})

    print(f"\n{dev} · {args.mode} · {args.dtype} · cfm {args.cfm_steps} {args.label}")
    print(f"{'text':<7} {'words':>5} {'ttfa s':>7} {'rtf':>5} {'audio s':>7} {'t3 s':>6} {'s3 s':>6} {'VRAM GB':>8}")
    for r in rows:
        print(f"{r['text']:<7} {r['words']:>5} {r['ttfa']:>7.2f} {r['rtf']:>5.2f} {r['audio']:>7.1f} "
              f"{r['t3']:>6.2f} {r['s3']:>6.2f} {r.get('vram_reserved', 0):>8.2f}")
    parallel = []
    if args.mode == "stream" and args.parallel:
        for n in [int(x) for x in args.parallel.split(",")]:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            res = run_parallel(target, TEXTS["medium"], n)
            if torch.cuda.is_available():
                res["vram_reserved"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            parallel.append(res)
            print(f"parallel {n}: worst ttfa {res['ttfa_max']:.2f} s, worst rtf {res['rtf_max']:.2f}, "
                  f"worst buffer {res['min_buffer_min']}, VRAM {res.get('vram_reserved')} GB", flush=True)
    out = {"device": dev, "mode": args.mode, "dtype": args.dtype, "cfm_steps": args.cfm_steps, "label": args.label,
           "parallel": parallel,
           "dec_cond": args.dec_cond, "graphs": args.graphs, "fast_flow": args.fast_flow,
           "first_chunk": args.first_chunk, "chunk": args.chunk, "rows": rows,
           "when": time.strftime("%Y-%m-%d %H:%M")}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("JSON " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
