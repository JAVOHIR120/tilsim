"""Uzbek TTS engine: Chatterbox (English base) + the open Uzbek LoRA by UAzimov (MIT), merged at load time.

    E:/tools/voiceclone/.venv/Scripts/python.exe engine/uz_tts.py "Assalomu alaykum!" "2026-yil 14-noyabr."
    ... --file lines.txt --ref my_voice.wav --out-dir out/test --check heldout.wav

Each text goes through uznorm first (numbers -> words, ASCII apostrophes: the model was trained that way),
is cut into sentences, and each sentence is voiced in the reference speaker's voice.

How the model is rebuilt (checked against the training toolkit, gokhaneraslan/chatterbox-finetuning):
- T3 with the vocabulary grown 704 -> 2454 tokens; all other base weights from t3_cfg.safetensors.
- LoRA (r=128, alpha=256) folded into the weights: W += 2.0 * B @ A, for q/k/v/o/gate/up/down + spkr_enc.
- text_emb and text_head replaced whole by the adapter's trained copies; the adapter's tokenizer.json.
- A vocab of 2454 makes chatterbox think "multilingual" and switch on an alignment analyzer tuned for
  Resemble's own multilingual model. The training toolkit never used it, so it is kept off here.
No peft needed. Weights are copied tensor by tensor straight to the GPU, so RAM stays low.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uznorm import for_model, normalize, split_sentences  # noqa: E402

from chatterbox.models.s3gen import S3Gen  # noqa: E402
from chatterbox.models.t3 import T3  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config  # noqa: E402
from chatterbox.models.tokenizers import EnTokenizer  # noqa: E402
from chatterbox.models.voice_encoder import VoiceEncoder  # noqa: E402
from chatterbox.tts import ChatterboxTTS, Conditionals  # noqa: E402

import os  # noqa: E402

# The cloud copy (deploy/modal_app.py) points these at the weights baked into its image.
BASE_DIR = Path(os.environ.get("TILSIM_BASE_DIR", "E:/tools/voiceclone/ckpt"))
ADAPTER_DIR = Path(os.environ.get("TILSIM_ADAPTER_DIR", ROOT / "models" / "uzbek-chatterbox-lora"))
DEFAULT_REF = Path("E:/tools/voiceclone/work/ref.wav")
VOCAB = 2454
LORA_SCALE = 256 / 128  # lora_alpha / r (adapter_config.json)
# The adapter author's demo settings; repetition_penalty 1.2 cuts sentences short with this adapter.
PARAMS = {"temperature": 0.8, "exaggeration": 0.5, "cfg_weight": 0.5, "repetition_penalty": 2.0}


class UzT3Config(T3Config):
    @property
    def is_multilingual(self):  # see the module docstring
        return False


def _copy_weights(module: torch.nn.Module, path: Path, strict: bool = True, skip: frozenset = frozenset()):
    """Copy a safetensors file into an existing module one tensor at a time (no second copy of the model in RAM)."""
    target = module.state_dict()
    loaded, unexpected = set(), []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key in skip:
                continue
            if key not in target:
                unexpected.append(key)
                continue
            src = f.get_tensor(key)
            if src.shape != target[key].shape:
                raise ValueError(f"{path.name}: {key} has shape {tuple(src.shape)}, model wants {tuple(target[key].shape)}")
            with torch.no_grad():
                target[key].copy_(src)
            loaded.add(key)
    missing = [k for k in target if k not in loaded and k not in skip]
    if strict and (missing or unexpected):
        raise RuntimeError(f"{path.name}: missing {missing[:5]}… unexpected {unexpected[:5]}…")
    return missing, unexpected


def _apply_adapter(t3: T3, path: Path) -> tuple[int, int]:
    params = dict(t3.named_parameters())
    prefix = "base_model.model."
    merged = replaced = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            name = key.removeprefix(prefix)
            if ".lora_B." in name:
                continue
            with torch.no_grad():
                if ".lora_A." in name:
                    module = name.split(".lora_A.")[0]
                    weight = params[f"{module}.weight"]
                    a = f.get_tensor(key).to(weight.device)
                    b = f.get_tensor(f"{prefix}{module}.lora_B.weight").to(weight.device)
                    weight.add_(b @ a, alpha=LORA_SCALE)
                    merged += 1
                else:  # modules_to_save: text_emb / text_head, trained in full
                    src = f.get_tensor(key)
                    if params[name].shape != src.shape:
                        raise ValueError(f"adapter {name}: {tuple(src.shape)} vs {tuple(params[name].shape)}")
                    params[name].copy_(src)
                    replaced += 1
    if merged != 211 or replaced != 2:  # 30 layers x 7 projections + spkr_enc; text_emb + text_head
        raise RuntimeError(f"adapter merged {merged} LoRA pairs and replaced {replaced} layers; expected 211 and 2")
    return merged, replaced


def load_engine(device: str | None = None, base_dir: Path = BASE_DIR, adapter_dir: Path = ADAPTER_DIR) -> ChatterboxTTS:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.device(device):  # build T3 straight on the GPU: no 2 GB copy in RAM
        t3 = T3(UzT3Config(text_tokens_dict_size=VOCAB))
    _copy_weights(t3, base_dir / "t3_cfg.safetensors", skip=frozenset({"text_emb.weight", "text_head.weight"}))
    _apply_adapter(t3, adapter_dir / "adapter_model.safetensors")
    t3.to(device).eval()

    try:
        with torch.device(device):
            s3gen = S3Gen()
    except RuntimeError:  # some init code mixes CPU numpy buffers; build on CPU instead
        s3gen = S3Gen()
    _copy_weights(s3gen, base_dir / "s3gen.safetensors", strict=False)  # chatterbox loads it non-strict too
    s3gen.to(device).eval()

    ve = VoiceEncoder()
    _copy_weights(ve, base_dir / "ve.safetensors")
    ve.to(device).eval()

    tokenizer = EnTokenizer(str(adapter_dir / "tokenizer.json"))
    if tokenizer.tokenizer.get_vocab_size() != VOCAB:
        raise RuntimeError(f"tokenizer has {tokenizer.tokenizer.get_vocab_size()} tokens, expected {VOCAB}")
    conds_path = base_dir / "conds.pt"
    conds = Conditionals.load(conds_path, map_location=device).to(device) if conds_path.exists() else None
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


def trim(x: np.ndarray, sr: int, head_pad: float = 0.05, tail_pad: float = 0.15) -> np.ndarray:
    """Cut silence around the speech, and drop sounds left after the last word (breaths, gasps, coughs).

    The end is the last strong voiced frame, extended through the sound still attached to it (a final -sh, -q, -t),
    but not across a gap: anything after a >=60 ms gap (an in-breath, a gasp the model added) is dropped.
    A soft 40 ms fade-out avoids the "cut off abruptly" ending the Gemini judge heard (2026-11-15).
    """
    hop = int(sr * 0.01)
    n = (len(x) - hop) // hop
    if n <= 0:
        return x
    frames = x[:n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    peak = db.max()
    audible = db > max(peak - 45, -60)
    if not audible.any():
        return x
    first = int(np.argmax(audible))
    strong = np.where((db > peak - 30) & (zcr < 0.25))[0]
    last = int(strong[-1]) if len(strong) else int(np.where(audible)[0][-1])
    gap, i = 0, last
    while i + 1 < n and i - last < 25:  # follow the attached tail for at most 0.25 s
        i += 1
        if audible[i]:
            gap = 0
        else:
            gap += 1
            if gap >= 6:
                break
    tail_end = i - gap  # last audible frame still attached to the speech
    start = max(0, first * hop - int(head_pad * sr))
    end = min(len(x), (tail_end + 1) * hop + int(tail_pad * sr))
    y = x[start:end].copy()
    fade_in, fade_out = min(int(0.008 * sr), len(y) // 2), min(int(0.04 * sr), len(y) // 2)
    if fade_in:
        y[:fade_in] *= np.linspace(0, 1, fade_in)
    if fade_out:
        y[-fade_out:] *= np.linspace(1, 0, fade_out) ** 2
    return y


class Prefix:
    """A real recording of the speaker (and its exact transcript) that every sentence continues from.

    Why: the open Uzbek LoRA was trained (gokhaneraslan toolkit) with the first ~3 s of every clip used as the voice
    prompt AND left out of the loss, so the model never learned to start a sentence: junk words at the start and a
    swallowed first word (2026-11-14: first word right in only 69 % of takes). Here the speech sequence starts with
    ~5 s of the speaker's real speech tokens and the text with its transcript, exactly like training; the model then
    goes on with the new sentence, and only the new tokens are turned into audio. No cutting by guesswork.
    """

    def __init__(self, engine: ChatterboxTTS, wav_path: str | Path, end_sec: float, text: str):
        import librosa
        wav, _ = librosa.load(str(wav_path), sr=16000)
        tokens, _ = engine.s3gen.tokenizer.forward([wav[:int(end_sec * 16000)]])
        self.tokens = torch.atleast_2d(tokens)[0].to(engine.device)
        self.text = normalize(text)


@torch.inference_mode()
def _continue_tokens(t3: T3, t3_cond, text_tokens: torch.Tensor, prefix: torch.Tensor, max_new_tokens: int,
                     temperature: float, cfg_weight: float, repetition_penalty: float, min_p: float = 0.05,
                     top_p: float = 1.0) -> torch.Tensor:
    """chatterbox's T3.inference loop, with real speech tokens in front of the generated ones (and no attention maps)."""
    from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                        TopPLogitsWarper)
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

    initial = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial,
                                        cfg_weight=cfg_weight)
    model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head,
                                 alignment_stream_analyzer=None)
    device = embeds.device
    bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
    n_pre = prefix.numel()
    head = torch.cat([t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0),
                      t3.speech_emb(prefix.view(1, -1)) +
                      t3.speech_pos_emb.get_fixed_embedding(torch.arange(1, n_pre + 1, device=device))], dim=1)
    out = model(inputs_embeds=torch.cat([embeds, torch.cat([head, head])], dim=1), past_key_values=None,
                use_cache=True, output_attentions=False, output_hidden_states=True, return_dict=True)
    past = out.past_key_values
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    min_p_w, top_p_w = MinPLogitsWarper(min_p=min_p), TopPLogitsWarper(top_p=top_p)
    generated, predicted = bos.clone(), []
    for i in range(max_new_tokens):
        step = out.logits[:, -1, :]
        logits = step[0:1] + cfg_weight * (step[0:1] - step[1:2])
        logits = rep(generated, logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = top_p_w(generated, min_p_w(generated, logits))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        predicted.append(nxt)
        generated = torch.cat([generated, nxt], dim=1)
        if nxt.view(-1) == t3.hp.stop_speech_token:
            break
        emb = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(n_pre + i + 1)
        out = model(inputs_embeds=torch.cat([emb, emb]), past_key_values=past, output_attentions=False,
                    output_hidden_states=True, return_dict=True)
        past = out.past_key_values
    return torch.cat(predicted, dim=1)[0]


def generate_continuing(engine: ChatterboxTTS, prefix: Prefix, sentence: str, temperature: float, exaggeration: float,
                        cfg_weight: float, repetition_penalty: float) -> np.ndarray:
    import torch.nn.functional as F
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.tts import punc_norm

    cond = engine.conds.t3
    if exaggeration != float(cond.emotion_adv[0, 0, 0]):
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        engine.conds.t3 = cond = T3Cond(speaker_emb=cond.speaker_emb, cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
                                        emotion_adv=exaggeration * torch.ones(1, 1, 1)).to(device=engine.device)
    text = punc_norm(f"{prefix.text} {sentence}")
    tokens = engine.tokenizer.text_to_tokens(text).to(engine.device)
    tokens = torch.cat([tokens, tokens], dim=0)  # CFG pair
    tokens = F.pad(F.pad(tokens, (1, 0), value=engine.t3.hp.start_text_token), (0, 1), value=engine.t3.hp.stop_text_token)
    letters = sum(ch.isalpha() for ch in sentence)
    speech = _continue_tokens(engine.t3, cond, tokens, prefix.tokens, max_new_tokens=min(1000, 80 + 5 * letters),
                              temperature=temperature, cfg_weight=cfg_weight, repetition_penalty=repetition_penalty)
    speech = drop_invalid_tokens(speech)
    speech = speech[speech < 6561].to(engine.device)
    with torch.inference_mode():
        wav, _ = engine.s3gen.inference(speech_tokens=speech, ref_dict=engine.conds.gen)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return engine.watermarker.apply_watermark(wav, sample_rate=engine.sr)


def speak(engine: ChatterboxTTS, text: str, seed: int = 0, pause: float = 0.25, prefix: Prefix | None = None,
          **params) -> tuple[str, np.ndarray]:
    """Uzbek text -> (spoken form, 24 kHz float audio). Call engine.prepare_conditionals(ref) first.
    With a Prefix every sentence continues from the speaker's real speech (see Prefix)."""
    p = {**PARAMS, **params}
    spoken = normalize(text)
    pieces = []
    for i, sentence in enumerate(split_sentences(spoken, max_chars=200)):
        torch.manual_seed(seed + i)
        sentence = for_model(sentence)  # model-only respellings (uznorm/respell.py); `spoken` stays real Uzbek
        if prefix is not None:
            wav = generate_continuing(engine, prefix, sentence, **p)
        else:
            wav = engine.generate(sentence, **p).squeeze(0).cpu().numpy()
        pieces += [trim(wav, engine.sr), np.zeros(int(pause * engine.sr), np.float32)]
    return spoken, np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)


def speaker_similarity(engine: ChatterboxTTS, a: np.ndarray | str, b: np.ndarray | str) -> float:
    """Cosine of Chatterbox's speaker embeddings, same method as E:/tools/voiceclone/similarity.py."""
    import librosa

    def emb(x):
        wav = librosa.load(x, sr=16000)[0] if isinstance(x, (str, Path)) else librosa.resample(x, orig_sr=engine.sr, target_sr=16000)
        with torch.inference_mode():
            e = np.asarray(engine.ve.embeds_from_wavs([wav], sample_rate=16000)).reshape(-1)
        return e / np.linalg.norm(e)

    return float(np.dot(emb(a), emb(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*", help="texts to say (or use --file / --tsv)")
    ap.add_argument("--file", type=Path, help="one text per line; lines starting with # are skipped")
    ap.add_argument("--tsv", type=Path, help="id<TAB>category<TAB>text lines (data/eval/test_uz.tsv)")
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0,1,2,3: several takes per line for best-of-N")
    ap.add_argument("--voice", default="ovozim", help="profile in data/voices/<name>.json (ref clip, prefix, settings); "
                    "flags given on the command line win. Use --voice none for bare defaults")
    ap.add_argument("--ref", type=Path, default=None, help="5-15 s of clean speech in the target voice")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--check", type=Path, help="held-out clip of the same speaker: prints voice similarity per file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep takes already in OUT/manifest.json, make only the rest")
    ap.add_argument("--prompt-sec", type=float, default=None, help="seconds of the reference used as the T3 speech prompt "
                    "(chatterbox default 6; the Uzbek LoRA was trained with 3; 6 tested better with a prefix)")
    ap.add_argument("--prefix-text", help="exact words spoken at the start of --ref: continue every sentence from them")
    ap.add_argument("--prefix-end", type=float, help="where that speech ends in --ref, seconds (include ~0.25 s pause)")
    ap.add_argument("--cpu", action="store_true")
    for name in PARAMS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=None)
    args = ap.parse_args()

    profile_path = ROOT / "data" / "voices" / f"{args.voice}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if args.voice != "none" and profile_path.exists() else {}
    if args.ref is None:
        args.ref = Path(profile.get("ref", DEFAULT_REF))
    if args.check is None and profile.get("check"):
        args.check = Path(profile["check"])
    if args.prefix_text is None and profile.get("prefix_text"):
        args.prefix_text, args.prefix_end = profile["prefix_text"], profile["prefix_end"]
    if args.prompt_sec is None:
        args.prompt_sec = float(profile.get("prompt_sec", 6.0))
    for name, value in {**PARAMS, **profile.get("params", {})}.items():
        if getattr(args, name) is None:
            setattr(args, name, float(value))

    items = [{"id": f"{n:02d}", "category": "", "text": t} for n, t in enumerate(args.texts, 1)]
    if args.file:
        lines = [ln.strip() for ln in args.file.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        items += [{"id": f"{len(items) + n:02d}", "category": "", "text": t} for n, t in enumerate(lines, 1)]
    if args.tsv:
        for ln in args.tsv.read_text(encoding="utf-8").splitlines():
            if ln.strip() and not ln.startswith("#"):
                item_id, category, text = ln.split("\t", 2)
                items.append({"id": item_id, "category": category, "text": text})
    if not items:
        ap.error("give texts, --file or --tsv")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    import torchaudio

    t0 = time.time()
    engine = load_engine("cpu" if args.cpu else None)
    vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and not args.cpu else 0
    print(f"model ready on {engine.device} in {time.time() - t0:.0f} s (VRAM {vram:.2f} GB)", flush=True)
    params = {k: getattr(args, k) for k in PARAMS}
    engine.ENC_COND_LEN = int(args.prompt_sec * 16000)  # instance override of the class constant (6 s at 16 kHz)
    engine.prepare_conditionals(str(args.ref), exaggeration=params["exaggeration"])
    prefix = Prefix(engine, args.ref, args.prefix_end, args.prefix_text) if args.prefix_text else None
    if prefix is not None:
        print(f"continuing from {prefix.tokens.numel()} real speech tokens: «{prefix.text}»", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    rows = []
    if args.resume and manifest_path.exists():
        rows = [r for r in json.loads(manifest_path.read_text(encoding="utf-8"))["lines"] if (args.out_dir / r["file"]).exists()]
    done = {r["file"] for r in rows}

    def write_manifest():  # after every take, so a stopped run loses nothing
        manifest_path.write_text(json.dumps({"ref": str(args.ref), "params": params, "seeds": seeds,
                                             "prompt_sec": args.prompt_sec, "prefix_text": args.prefix_text,
                                             "prefix_end": args.prefix_end, "lines": rows},
                                            ensure_ascii=False, indent=1), encoding="utf-8")

    for item in items:
        for seed in seeds:
            name = f"{item['id']}.wav" if len(seeds) == 1 else f"{item['id']}__s{seed}.wav"
            if name in done:
                continue
            t1 = time.time()
            spoken, audio = speak(engine, item["text"], seed=seed, prefix=prefix, **params)
            took = time.time() - t1
            path = args.out_dir / name
            torchaudio.save(str(path), torch.from_numpy(audio).unsqueeze(0), engine.sr)
            row = {"file": path.name, "id": item["id"], "category": item["category"], "seed": seed,
                   "text": item["text"], "spoken": spoken, "seconds": round(len(audio) / engine.sr, 2),
                   "gen_seconds": round(took, 1), "rtf": round(took / max(len(audio) / engine.sr, 1e-6), 2)}
            if args.check:
                row["similarity"] = round(speaker_similarity(engine, str(args.check), audio), 3)
            rows.append(row)
            write_manifest()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if torch.cuda.is_available() and not args.cpu:
                # 4 GB card: without this, cached blocks pile up over ~100 takes, the driver spills into
                # shared system RAM and each take gets ~7x slower (seen 2026-11-14: 5 s -> 35 s per take)
                torch.cuda.empty_cache()
    if torch.cuda.is_available() and not args.cpu:
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    write_manifest()
    print(f"saved {len(rows)} files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
