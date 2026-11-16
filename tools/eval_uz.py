"""Listen to generated Uzbek audio with machines: pronunciation (ASR round-trip), naturalness (UTMOS), voice similarity.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/eval_uz.py out/eval-base [--ref E:/tools/voiceclone/work/ref.wav]

Reads OUT/manifest.json (from engine/uz_tts.py), writes OUT/eval.json and prints a summary.

- Pronunciation: the Uzbek Whisper model (Kotib/uzbek_stt_v1, Apache-2.0) writes down what it hears; both texts go
  through uznorm and are compared letter by letter (CER) and word by word (WER). Uzbek-specific letters are aligned
  one by one, so «o' heard as o», «q heard as k», «tutuq dropped» are counted separately.
  Whisper tends to "fix" sloppy speech into real words, so these numbers are a lower bound on real errors.
- Naturalness: UTMOS22 strong (MIT), 1-5. Trained on English TTS ratings: use it to compare takes, not as an absolute.
- Similarity: cosine of Chatterbox's speaker embedding against --ref (same as E:/tools/voiceclone/similarity.py).
Stages run one after another (GPU for ASR, CPU for the rest), so it fits on the 4 GB card.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uznorm import normalize  # noqa: E402

ASR_DIR = ROOT / "models" / "asr-kotib-whisper-medium-uz"
ASR_DIR_EN = ROOT / "models" / "asr-whisper-large-v3-turbo"   # openai/whisper-large-v3-turbo, MIT (English mode)
MOS_DIR = ROOT / "models" / "speechmos"
VE_PATH = Path("E:/tools/voiceclone/ckpt/ve.safetensors")
DEFAULT_REF = Path("E:/tools/voiceclone/work/ref.wav")
FOCUS = ["o'", "g'", "q", "x", "h", "'", "sh", "ch", "o", "g", "k"]
UNIT_RE = re.compile(r"o'|g'|sh|ch|'|[a-z]")


# ---------------------------------------------------------------- text comparison

SAME_WORD = [(r"sentyabr", "sentabr"), (r"oktyabr", "oktabr")]  # spelling vs. how people say it: not an error


def canon(text: str) -> str:
    """Spoken form, lowercase, ASCII apostrophe, letters and spaces only."""
    t = normalize(text).lower()
    for said, written in SAME_WORD:
        t = re.sub(said, written, t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])(?<![og]')", " ", t)  # apostrophes that are not o'/g'/tutuq
    return re.sub(r"\s+", " ", t).strip()


def align(ref: list, hyp: list) -> list[tuple]:
    """Levenshtein alignment -> list of (ref_item | None, hyp_item | None)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def errors(pairs: list[tuple]) -> int:
    return sum(r != h for r, h in pairs)


def compare(ref_text: str, hyp_text: str) -> dict:
    ref, hyp = canon(ref_text), canon(hyp_text)
    ref_chars, hyp_chars = list(ref), list(hyp)
    ref_words, hyp_words = ref.split(), hyp.split()
    units = align([u for w in ref_words for u in UNIT_RE.findall(w) + [" "]],
                  [u for w in hyp_words for u in UNIT_RE.findall(w) + [" "]])
    confusions = [(r, h or "∅") for r, h in units if r in FOCUS and r != h]
    focus_total = Counter(r for r, _ in units if r in FOCUS)
    word_pairs = align(ref_words, hyp_words)
    lead_junk = 0
    for r, h in word_pairs:  # words heard before the first word of the text
        if r is not None:
            break
        lead_junk += 1
    first_ok = next(((r == h) for r, h in word_pairs if r is not None), False)
    return {
        "lead_junk": lead_junk, "first_word_ok": first_ok,
        "ref": ref, "hyp": hyp,
        "cer": round(errors(align(ref_chars, hyp_chars)) / max(len(ref_chars), 1), 4),
        "wer": round(errors(word_pairs) / max(len(ref_words), 1), 4),
        "wrong_words": [f"{r or '∅'}→{h or '∅'}" for r, h in word_pairs if r != h],
        "confusions": [f"{r}→{h}" for r, h in confusions],
        "focus_total": dict(focus_total),
    }


_SPELLING: dict | None = None


def canon_en(text: str) -> str:
    """English text -> spoken words, lowercase, American spelling, letters/apostrophes/spaces only.
    Digits Whisper writes ("9:30", "$45", "2003") become the same words the voice was given (uznorm.en_normalize)."""
    global _SPELLING
    from uznorm.en_normalize import normalize_en
    if _SPELLING is None:
        path = ASR_DIR_EN / "normalizer.json"   # Whisper's British -> American spelling map
        _SPELLING = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    t = normalize_en(text).lower().replace("’", "'")
    t = re.sub(r"[-–—/]", " ", t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return " ".join(_SPELLING.get(w, w) for w in t.split())


def compare_en(ref_text: str, hyp_text: str) -> dict:
    ref, hyp = canon_en(ref_text), canon_en(hyp_text)
    word_pairs = align(ref.split(), hyp.split())
    lead_junk = 0
    for r, h in word_pairs:
        if r is not None:
            break
        lead_junk += 1
    first_ok = next(((r == h) for r, h in word_pairs if r is not None), False)
    return {"lead_junk": lead_junk, "first_word_ok": first_ok, "ref": ref, "hyp": hyp,
            "cer": round(errors(align(list(ref), list(hyp))) / max(len(ref), 1), 4),
            "wer": round(errors(word_pairs) / max(len(ref.split()), 1), 4),
            "wrong_words": [f"{r or '∅'}→{h or '∅'}" for r, h in word_pairs if r != h]}


# ---------------------------------------------------------------- stages

def run_asr(paths: list[Path], batch: int = 1, lang: str = "uz") -> list[str]:
    # batch 1 on purpose: Whisper keeps a cross-attention cache per beam for the full 30 s window
    # (24 layers x 1500 frames). 6 files x 5 beams needed ~4.4 GB and spilled out of the 4 GB card (10 s per file).
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    asr_dir = ASR_DIR_EN if lang == "en" else ASR_DIR
    proc = WhisperProcessor.from_pretrained(asr_dir)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(asr_dir, dtype=torch.float16, device_map="cuda")
    except Exception:  # older/newer transformers or no accelerate: load then move
        model = WhisperForConditionalGeneration.from_pretrained(asr_dir, dtype=torch.float16).to("cuda")
    model.eval()
    out = []
    for i in range(0, len(paths), batch):
        audio = [librosa.load(p, sr=16000)[0] for p in paths[i:i + batch]]
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.inference_mode():
            ids = model.generate(feats, language=lang, task="transcribe", num_beams=5, max_new_tokens=200)
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        if (i // batch) % 12 == 0 or i + batch >= len(paths):
            vram = torch.cuda.max_memory_allocated() / 2**30
            print(f"  asr {min(i + batch, len(paths))}/{len(paths)}  peak VRAM {vram:.2f} GB", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def utmos_weights() -> dict:
    """SpeechMOS checkpoint; if absent, the same UTMOS22-strong weights from Blinorot/UTMOS-PyTorch with renamed keys
    (domain row 0 and listener 288, the "mean judge" the original UTMOS22 demo uses)."""
    official = MOS_DIR / "utmos22_strong_step7459_v1.pt"
    if official.exists() and official.stat().st_size == 411_179_119:
        return torch.load(official, map_location="cpu", weights_only=True)
    sd = torch.load(MOS_DIR / "blinorot_utmos_state_dict.pt", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    out = {}
    for k, v in sd.items():
        if k.startswith("feature_extractors.0.ssl_model."):
            out["wav2vec2." + k.removeprefix("feature_extractors.0.ssl_model.")] = v
        elif k.startswith("output_layers.0.decoder_rnn."):
            out["blstm." + k.removeprefix("output_layers.0.decoder_rnn.")] = v
    out["domain_emb"] = sd["feature_extractors.1.embedding.weight"][0:1]
    out["judge_emb"] = sd["output_layers.0.judge_embedding.weight"][288:289]
    for i_theirs, i_ours in ((0, 0), (3, 2)):
        for part in ("weight", "bias"):
            out[f"projection.{i_ours}.{part}"] = sd[f"output_layers.1.net.{i_theirs}.{part}"]
    return out


def run_mos(paths: list[Path]) -> list[float]:
    sys.path.insert(0, str(MOS_DIR))
    from speechmos.utmos22.strong.model import UTMOS22Strong

    model = UTMOS22Strong()
    model.load_state_dict(utmos_weights())  # strict: every key must match
    model.eval()
    scores = []
    for p in paths:
        wav, _ = librosa.load(p, sr=16000)
        with torch.inference_mode():
            scores.append(round(float(model(torch.from_numpy(wav).unsqueeze(0), sr=16000)), 3))
    del model
    gc.collect()
    return scores


def run_similarity(paths: list[Path], ref: Path) -> list[float]:
    from safetensors.torch import load_file
    from chatterbox.models.voice_encoder import VoiceEncoder

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(VE_PATH))
    ve.eval()

    def emb(p):
        wav, _ = librosa.load(p, sr=16000)
        with torch.inference_mode():
            e = np.asarray(ve.embeds_from_wavs([wav], sample_rate=16000)).reshape(-1)
        return e / np.linalg.norm(e)

    r = emb(ref)
    return [round(float(np.dot(r, emb(p))), 4) for p in paths]


PROFILE = ROOT / "data" / "eval" / "speaker_profile.json"


def run_prosody(rows: list[dict], paths: list[Path]) -> None:
    """Pitch spread + speaking rate per take, and a penalty for being flatter or wilder than the real speaker."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prosody import features, penalty

    real = json.loads(PROFILE.read_text(encoding="utf-8"))
    for r, p in zip(rows, paths):
        f = features(p, r.get("spoken") or r.get("text"))
        r.update({"pitch_st": f["pitch_st"], "rate": f.get("rate"), "prosody_penalty": penalty(f, real),
                  "pitch_hz": f.get("pitch_median_hz"), "pitch_range_st": f.get("pitch_range_st"), "loud_db": f.get("loud_db")})
        style = "flat" if f["pitch_st"] < real["pitch_st"] * 0.75 else "wild" if f["pitch_st"] > real["pitch_st"] * 1.35 else "ok"
        r["prosody"] = style


# ---------------------------------------------------------------- report

def summarize(rows: list[dict]) -> dict:
    def mean(key, subset=rows):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    conf = Counter(c for r in rows for c in r.get("confusions", []))
    totals = Counter()
    for r in rows:
        totals.update(r.get("focus_total", {}))
    wrong = Counter()
    for c, n in conf.items():
        wrong[c.split("→")[0]] += n
    letters = {u: {"n": totals[u], "ok": round(1 - wrong[u] / totals[u], 3)} for u in FOCUS if totals[u]}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("category") or "-"].append(r)
    with_asr = [r for r in rows if "lead_junk" in r]
    return {
        "files": len(rows), "cer": mean("cer"), "wer": mean("wer"), "utmos": mean("utmos"), "similarity": mean("similarity"),
        "junk_at_start": round(sum(r["lead_junk"] > 0 for r in with_asr) / len(with_asr), 3) if with_asr else None,
        "first_word_ok": round(sum(r["first_word_ok"] for r in with_asr) / len(with_asr), 3) if with_asr else None,
        "prosody": dict(Counter(r["prosody"] for r in rows if "prosody" in r)),
        "prosody_penalty": mean("prosody_penalty"), "pitch_st": mean("pitch_st"), "rate": mean("rate"),
        "letters_correct": letters, "top_confusions": conf.most_common(15),
        "by_category": {k: {"cer": mean("cer", v), "utmos": mean("utmos", v), "n": len(v)} for k, v in sorted(by_cat.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="folder with manifest.json")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-mos", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-prosody", action="store_true")
    ap.add_argument("--only-prosody", action="store_true", help="add prosody to an existing eval.json, run nothing else")
    ap.add_argument("--lang", choices=["uz", "en"], default="uz", help="en: English Whisper and English text rules "
                    "(no Uzbek letter stats, no Uzbek prosody profile)")
    args = ap.parse_args()

    manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["lines"]
    if args.only_prosody:
        rows = json.loads((args.out / "eval.json").read_text(encoding="utf-8"))["lines"]
        args.no_asr = args.no_mos = args.no_sim = True
    paths = [args.out / r["file"] for r in rows]
    if not args.no_asr:
        print("ASR …", flush=True)
        for r, hyp in zip(rows, run_asr(paths, lang=args.lang)):
            r["asr"] = hyp
            r.update(compare_en(r["text"], hyp) if args.lang == "en" else compare(r["text"], hyp))
    if not args.no_mos:
        print("UTMOS …", flush=True)
        for r, s in zip(rows, run_mos(paths)):
            r["utmos"] = s
    if not args.no_sim:
        print("similarity …", flush=True)
        for r, s in zip(rows, run_similarity(paths, args.ref)):
            r["similarity"] = s
    if not args.no_prosody and args.lang == "uz":   # the speaker profile is Uzbek speech
        print("prosody …", flush=True)
        run_prosody(rows, paths)
    summary = summarize(rows)
    (args.out / "eval.json").write_text(json.dumps({"summary": summary, "lines": rows}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    worst = sorted((r for r in rows if "cer" in r), key=lambda r: -r["cer"])[:8]
    print("\nworst lines:")
    for r in worst:
        print(f"  {r['file']}  CER {r['cer']:.3f}  {r['ref']}\n{'':>14}heard: {r['hyp']}")


if __name__ == "__main__":
    main()
