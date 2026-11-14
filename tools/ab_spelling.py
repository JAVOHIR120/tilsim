"""Which way of WRITING a word makes the voice SAY it right? Generate each spelling, then let the ASR choose.

A plain ASR transcript hides vowel errors (Whisper writes «noyabr» even when it hears «noyebr»), so this uses a
forced choice instead: for each take, Whisper scores the right word against the wrong-sounding word in the same
sentence; margin = log P(right) - log P(wrong). Bigger margin = the vowel sounded more like it should.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/ab_spelling.py --out out/ab-ya --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "tools"))

# word: (sentence template, spellings to try, right-sounding forms, wrong-sounding forms)
SETS = {}
SETS["months"] = TESTS = {
    "noyabr": ("O'n beshinchi {w} kuni uchrashamiz.", ["noyabr", "noyaabr", "no-yabr", "noyobr", "noiyabr"],
               ["noyabr"], ["noyebr", "noyibr"]),
    "sentabr": ("Birinchi {w} kuni maktabga boramiz.", ["sentabr", "sentyabr", "sentaabr", "sen-tabr"],
                ["sentabr", "sentyabr"], ["sentebr", "sentyebr", "sentibr"]),
    "oktabr": ("{W} oyida havo ancha soviydi.", ["oktabr", "oktyabr", "oktaabr", "ok-tabr"],
               ["oktabr", "oktyabr"], ["oktebr", "oktyebr", "oktibr"]),
    "dekabr": ("{W} oxirida qor yog'adi.", ["dekabr", "dekaabr"], ["dekabr"], ["dekebr", "dekibr"]),
    "yangi": ("{W} kun boshlandi.", ["yangi", "yaangi"], ["yangi"], ["yengi"]),
    "yaxshi": ("Hammasi {w} bo'ladi.", ["yaxshi", "yaaxshi"], ["yaxshi"], ["yexshi"]),
}
# Does «ya» -> «yaa» help every word that starts with «ya»? (the owner chose it for yangi and yaxshi)
SETS["ya_general"] = {
    w: (tpl, [w, "yaa" + w[2:]], [w], ["ye" + w[2:]])
    for w, tpl in [
        ("yana", "U {w} keldi."), ("yaqin", "Bekat juda {w} joyda."), ("yarim", "Soat o'n ikki {w} bo'ldi."),
        ("yashil", "Bahorda daraxtlar {w} bo'ladi."), ("yagona", "Bu bizning {w} imkoniyatimiz."),
        ("yakshanba", "{W} kuni dam olamiz."), ("yashaydi", "U Toshkentda {w}."), ("yaratdi", "Usta chiroyli buyum {w}."),
    ]
}


def generate(out: Path, seeds: list[int]) -> list[dict]:
    from uz_tts import Prefix, generate_continuing, load_engine, trim
    import torchaudio

    profile = json.loads((ROOT / "data/voices/ovozim.json").read_text(encoding="utf-8"))
    engine = load_engine()
    engine.ENC_COND_LEN = int(profile["prompt_sec"] * 16000)
    p = profile["params"]
    engine.prepare_conditionals(profile["ref"], exaggeration=p["exaggeration"])
    prefix = Prefix(engine, profile["ref"], profile["prefix_end"], profile["prefix_text"])
    rows = []
    for word, (template, spellings, right, wrong) in TESTS.items():
        for spelling in spellings:
            sentence = template.format(w=spelling, W=spelling[0].upper() + spelling[1:])
            for seed in seeds:
                torch.manual_seed(seed)
                wav = trim(generate_continuing(engine, prefix, sentence, **p), engine.sr)  # no uznorm: spelling as is
                name = f"{word}__{spelling.replace('-', '_')}__s{seed}.wav"
                torchaudio.save(str(out / name), torch.from_numpy(wav).unsqueeze(0), engine.sr)
                rows.append({"file": name, "word": word, "spelling": spelling, "seed": seed, "sentence": sentence})
                print("made", name, flush=True)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def score(out: Path, rows: list[dict]) -> None:
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from eval_uz import ASR_DIR

    proc = WhisperProcessor.from_pretrained(ASR_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_DIR, dtype=torch.float16).to("cuda").eval()
    tok = proc.tokenizer
    tok.set_prefix_tokens(language="uzbek", task="transcribe", predict_timestamps=False)

    def logp(feats, text: str) -> float:
        ids = torch.tensor([tok(text).input_ids], device="cuda")
        with torch.inference_mode():
            logits = model(input_features=feats, decoder_input_ids=ids[:, :-1]).logits.float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None])[0, :, 0]
        return float(lp[3:].sum())  # skip the language/task/notimestamps prefix positions

    for r in rows:
        template, _, right, wrong = TESTS[r["word"]]
        audio, _ = librosa.load(out / r["file"], sr=16000)
        feats = proc([audio], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        fill = lambda w: template.format(w=w, W=w[0].upper() + w[1:])  # noqa: E731
        best_right = max(logp(feats, fill(w)) for w in right)
        best_wrong = max(logp(feats, fill(w)) for w in wrong)
        r["margin"] = round(best_right - best_wrong, 2)
        with torch.inference_mode():
            ids = model.generate(feats, language="uz", task="transcribe", num_beams=5, max_new_tokens=60)
        r["heard"] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(f"{r['file']:34} margin {r['margin']:6.2f}  heard: {r['heard']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", default="months", choices=sorted(SETS))
    args = ap.parse_args()
    global TESTS
    TESTS = SETS[args.set]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = generate(args.out, [int(s) for s in args.seeds.split(",")])
    score(args.out, rows)
    (args.out / "ab.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nmean margin by spelling (higher = the vowel sounds right):")
    for word in TESTS:
        for spelling in TESTS[word][1]:
            m = [r["margin"] for r in rows if r["word"] == word and r["spelling"] == spelling]
            print(f"  {word:8} {spelling:10} {np.mean(m):6.2f}   (takes: {', '.join(f'{x:.1f}' for x in m)})")


if __name__ == "__main__":
    main()
