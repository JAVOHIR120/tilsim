"""English-mode lab: the founder's voice reading English with the stock Chatterbox English T3 (no Uzbek LoRA).

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/en_lab.py --tsv data/eval/test_en.tsv --out-dir out/en-lab/E1
    ... --spkr uz                       # LoRA'd speaker projection instead of the stock one
    ... --prompt-wav clip.wav           # T3 voice prompt (speech tokens) from another clip than the reference
    ... --prefix-wav clip.wav --prefix-text "exact words" --prefix-end 4.2   # continue from real (or chosen) speech
    ... --exaggeration 0.4 --cfg-weight 0.3 --temperature 0.7

Writes OUT/manifest.json in engine/uz_tts.py's format, so tools/eval_uz.py --lang en and
tools/gemini_judge.py --mode en read it unchanged. Text goes through uznorm.en_normalize (English numbers),
never through the Uzbek normaliser or the English respelling.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))
import uz_tts  # noqa: E402
from uznorm.en_normalize import normalize_en  # noqa: E402

STOCK = {"temperature": 0.8, "exaggeration": 0.5, "cfg_weight": 0.5, "repetition_penalty": 1.2}


def load(spkr: str, ref: str, prompt_wav: str | None, prompt_sec: float, dtype: str = "bf16",
         t3_prompt: str = "ref", t3_spk: str = "ref", spk_mix: float = 0.0, spk_npy: str | None = None):
    import librosa
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.tts import ChatterboxTTS

    device = "cuda"
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    t3, tok = uz_tts.load_english_t3(device, spkr=spkr, dtype=dt)
    s3gen = S3Gen()
    uz_tts._copy_weights(s3gen, uz_tts.BASE_DIR / "s3gen.safetensors", strict=False)
    s3gen.to(device).eval()
    ve = VoiceEncoder()
    uz_tts._copy_weights(ve, uz_tts.BASE_DIR / "ve.safetensors")
    ve.to(device).eval()
    eng = ChatterboxTTS(t3, s3gen, ve, tok, device)
    eng.ENC_COND_LEN = int(prompt_sec * 16000)
    eng.prepare_conditionals(ref, exaggeration=0.5)
    if prompt_wav:  # T3's speech-token voice prompt from a different clip (speaker embedding stays from ref)
        wav = librosa.load(prompt_wav, sr=16000)[0][:eng.ENC_COND_LEN]
        toks, _ = s3gen.tokenizer.forward([wav], max_len=t3.hp.speech_cond_prompt_len)
        eng.conds.t3.cond_prompt_speech_tokens = torch.atleast_2d(toks).to(device)
        eng.conds.t3.cond_prompt_speech_emb = None
    # T3's "how to speak" prompt and speaker embedding can come from Chatterbox's built-in English voice (conds.pt,
    # shipped with the MIT model): the accent then comes from a native speaker, the timbre still from S3Gen's ref.
    if "conds" in (t3_prompt, t3_spk) or spk_mix > 0:
        from chatterbox.tts import Conditionals
        dc = Conditionals.load(uz_tts.BASE_DIR / "conds.pt", map_location=device).t3
        if t3_prompt == "conds":
            eng.conds.t3.cond_prompt_speech_tokens = dc.cond_prompt_speech_tokens.to(device)
            eng.conds.t3.cond_prompt_speech_emb = None
        if t3_spk == "conds":
            eng.conds.t3.speaker_emb = dc.speaker_emb.to(device)
        elif spk_mix > 0:   # between the founder (0) and a native voice (1), on the unit sphere
            a = eng.conds.t3.speaker_emb.float()
            b = (torch.from_numpy(__import__("numpy").load(spk_npy)).view(1, -1).to(device).float() if spk_npy
                 else dc.speaker_emb.to(device).float())
            mix = (1 - spk_mix) * a / a.norm() + spk_mix * b / b.norm()
            eng.conds.t3.speaker_emb = mix / mix.norm() * a.norm()
    if t3_prompt == "none":   # no speech prompt at all: speaker embedding only
        eng.conds.t3.cond_prompt_speech_tokens = None
        eng.conds.t3.cond_prompt_speech_emb = None
    eng.conds.t3.to(dtype=dt)
    return eng


def speak_en(eng, sentence: str, seed: int, prefix, p: dict):
    import torch.nn.functional as F
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.tts import punc_norm

    torch.manual_seed(seed)
    c = eng.conds.t3
    cond = T3Cond(speaker_emb=c.speaker_emb, cond_prompt_speech_tokens=c.cond_prompt_speech_tokens,
                  emotion_adv=p["exaggeration"] * torch.ones(1, 1, 1)).to(device=eng.device, dtype=c.speaker_emb.dtype)
    text = punc_norm(f"{prefix.text} {sentence}" if prefix else sentence)
    tokens = eng.tokenizer.text_to_tokens(text).to(eng.device)
    tokens = torch.cat([tokens, tokens], dim=0)
    tokens = F.pad(F.pad(tokens, (1, 0), value=eng.t3.hp.start_text_token), (0, 1), value=eng.t3.hp.stop_text_token)
    letters = sum(ch.isalpha() for ch in sentence)
    speech = uz_tts._continue_tokens(eng.t3, cond, tokens, prefix.tokens if prefix else None,
                                     max_new_tokens=min(1000, 80 + 5 * letters), temperature=p["temperature"],
                                     cfg_weight=p["cfg_weight"], repetition_penalty=p["repetition_penalty"])
    speech = drop_invalid_tokens(speech)
    speech = speech[speech < 6561].to(eng.device)
    with torch.inference_mode():
        wav, _ = eng.s3gen.inference(speech_tokens=speech, ref_dict=eng.conds.gen)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return uz_tts.trim(eng.watermarker.apply_watermark(wav, sample_rate=eng.sr), eng.sr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ids", default="", help="only these ids (comma list)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--spkr", choices=["base", "uz"], default="base")
    ap.add_argument("--ref", default="E:/tools/voiceclone/work/ref.wav", help="speaker embedding + S3Gen timbre")
    ap.add_argument("--check", default="E:/tools/voiceclone/work/heldout.wav")
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--prompt-sec", type=float, default=6.0)
    ap.add_argument("--prefix-wav", default=None)
    ap.add_argument("--prefix-text", default=None)
    ap.add_argument("--prefix-end", type=float, default=None)
    ap.add_argument("--t3-prompt", choices=["ref", "conds", "none"], default="ref",
                    help="T3 speech prompt: from --ref/--prompt-wav, from Chatterbox's built-in voice, or none")
    ap.add_argument("--t3-spk", choices=["ref", "conds"], default="ref", help="T3 speaker embedding source")
    ap.add_argument("--t3-spk-npy", default=None, help="native voice embedding (.npy, VoiceEncoder) to mix towards "
                    "instead of the built-in voice")
    ap.add_argument("--t3-spk-mix", type=float, default=0.0,
                    help="0..1: T3 speaker embedding this far from the founder towards the built-in native voice")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="T3 precision; fp32 does not fit next to S3Gen on the 4 GB laptop without spilling")
    for k, v in STOCK.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args()
    p = {k: getattr(args, k) for k in STOCK}
    ids = set(args.ids.split(",")) if args.ids else None
    items = []
    for ln in args.tsv.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            i, cat, text = ln.split("\t", 2)
            if ids is None or i in ids:
                items.append({"id": i, "category": cat, "text": text})
    seeds = [int(s) for s in args.seeds.split(",")]

    import torchaudio
    t0 = time.time()
    eng = load(args.spkr, args.ref, args.prompt_wav, args.prompt_sec, args.dtype, args.t3_prompt, args.t3_spk,
               args.t3_spk_mix, args.t3_spk_npy)
    prefix = None
    if args.prefix_wav:
        prefix = uz_tts.Prefix(eng, args.prefix_wav, args.prefix_end, args.prefix_text)
        prefix.text = args.prefix_text  # English: no Uzbek normalising
    print(f"ready in {time.time() - t0:.0f} s, VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        for seed in seeds:
            name = f"{it['id']}.wav" if len(seeds) == 1 else f"{it['id']}__s{seed}.wav"
            spoken = normalize_en(it["text"])
            t1 = time.time()
            audio = speak_en(eng, spoken, seed, prefix, p)
            took = time.time() - t1
            torchaudio.save(str(args.out_dir / name), torch.from_numpy(audio).unsqueeze(0), eng.sr)
            row = {"file": name, "id": it["id"], "category": it["category"], "seed": seed, "text": it["text"],
                   "spoken": spoken, "seconds": round(len(audio) / eng.sr, 2), "gen_seconds": round(took, 1),
                   "similarity": round(uz_tts.speaker_similarity(eng, args.check, audio), 3)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            torch.cuda.empty_cache()
            (args.out_dir / "manifest.json").write_text(json.dumps(
                {"variant": vars(args) | {"tsv": str(args.tsv), "out_dir": str(args.out_dir)}, "params": p,
                 "lines": rows}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; {len(rows)} files in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
