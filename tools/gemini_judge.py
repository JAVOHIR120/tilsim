"""A native-ear judge with a hard spending cap: gemini-3.8-flash rates each clip ALONE (no comparisons, no order bias).

    python tools/gemini_judge.py out/v2-calm/best out/bench-azure-sardor ... --ids t01,t02 --max-usd 0.40

- Uses GEMINI_API_KEY from E:/Ielts buddy/.env (never printed). That project is on a paid tier: every call costs money,
  so spend is computed from each response's own token counts (paid prices through 2026-12-31: input $0.75/M,
  output incl. thinking $3.75/M) and written to out/gemini_spend.json. The run stops before the total passes --max-usd.
- Prompt adapted from the owner's videos/imtihon-story/scripts/eval.py (native + delivery, 1-10, concrete issues).
- Results: <folder>/gemini.json, keyed by file name; cached, so a re-run never pays twice for the same clip.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ENV = Path("E:/Ielts buddy/.env")
MODEL = "gemini-3.8-flash"
PRICE_IN, PRICE_OUT = 0.75e-6, 3.75e-6
LEDGER = ROOT / "out" / "gemini_spend.json"

EAR = """You are a strict native Uzbek (Tashkent) speech expert. Listen to this ONE audio clip of Uzbek speech.
The intended text: «{say}»
Judge only the audio, as if hearing it on the radio. Rate:
- native: 1-10, how convincingly it sounds like a native Tashkent Uzbek speaker (vowels o/o', a after y, consonants q, g', x, h,
  word stress, sentence melody). 10 = indistinguishable from a native speaker; 5 = understandable but clearly foreign or synthetic.
- natural: 1-10, how human and natural it sounds (rhythm, pauses, intonation, not robotic, not exaggerated).
- issues: concrete problems you hear (the word + what is wrong), empty if none.
Answer JSON {{"native": n, "natural": n, "issues": [..]}}"""


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("GEMINI_API_KEY not found")


def ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"usd": 0.0, "calls": 0}


def call(key: str, parts: list) -> dict:
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (400, 403, 404):
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            time.sleep(5 * (attempt + 1) ** 2)
    raise RuntimeError("no answer")


def wav16(path: Path) -> str:
    y, _ = librosa.load(path, sr=16000)
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--ids", default="", help="only these ids (comma list); default all")
    ap.add_argument("--max-usd", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=None, help="only takes with this seed (e.g. 0 = first take, no selection)")
    args = ap.parse_args()
    key = api_key()
    ids = set(args.ids.split(",")) if args.ids else None
    spend = ledger()
    jobs = []
    for folder in args.folders:
        rows = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["lines"]
        cache_path = folder / "gemini.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        jobs += [(folder, cache_path, cache, r) for r in rows if (ids is None or r["id"] in ids) and r["file"] not in cache
                 and (args.seed is None or r.get("seed", 0) == args.seed)]
    # interleave folders so a stop at the cap still leaves every voice with some judged clips
    jobs.sort(key=lambda j: (j[3]["id"], str(j[0])))
    print(f"{len(jobs)} clips to judge; spent so far ${spend['usd']:.4f} of ${args.max_usd:.2f}", flush=True)
    for folder, cache_path, cache, r in jobs:
        avg = spend["usd"] / spend["calls"] if spend["calls"] else 0.003
        if spend["usd"] + max(avg * 2, 0.003) > args.max_usd:
            print(f"stopping: next call could pass ${args.max_usd:.2f} (spent ${spend['usd']:.4f})", flush=True)
            break
        audio = {"inlineData": {"mimeType": "audio/wav", "data": wav16(folder / r["file"])}}
        try:
            resp = call(key, [audio, {"text": EAR.format(say=r.get("spoken") or r["text"])}])
        except RuntimeError as e:
            print("API error, stopping:", e, flush=True)
            break
        u = resp.get("usageMetadata", {})
        cost = u.get("promptTokenCount", 0) * PRICE_IN + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * PRICE_OUT
        spend["usd"] = round(spend["usd"] + cost, 6)
        spend["calls"] += 1
        LEDGER.write_text(json.dumps(spend))
        txt = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if not p.get("thought"))
        try:
            verdict = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except json.JSONDecodeError:
            verdict = {"error": txt[:200]}
        cache[r["file"]] = {**verdict, "cost_usd": round(cost, 6)}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{folder.name:24} {r['file']:14} native {verdict.get('native')}  natural {verdict.get('natural')}  "
              f"${cost:.4f}  total ${spend['usd']:.4f}  {verdict.get('issues', '')}", flush=True)
    print(f"done; total spent ${spend['usd']:.4f} in {spend['calls']} calls", flush=True)


if __name__ == "__main__":
    main()
