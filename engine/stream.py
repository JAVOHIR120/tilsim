"""Live Uzbek speech: text in, 24 kHz audio chunks out, the first sound well under a second after the call.

    from stream import Streamer
    s = Streamer.load("ovozim", dtype="bf16")      # once, at server start
    s.warmup()
    for ev in s.stream("Salom! Endi Part 2 ga o‘tamiz.", cancel=event):
        ev["type"] == "mark"  -> {"text": original sentence, "index": i}
        ev["type"] == "audio" -> {"pcm": float32 numpy, 24 kHz mono}

How it works (the model is the same Chatterbox + Uzbek LoRA + Prefix continuation as engine/uz_tts.py):
1. The text is cut into sentences first. uznorm then normalises each sentence on its own, so a «mark» can name
   the sentence the listener is hearing.
2. T3 makes speech tokens one by one, like uz_tts._continue_tokens, but as a generator.
3. After the first FIRST_CHUNK tokens (plus S3Gen's 3-token look-ahead), and then every CHUNK tokens, S3Gen's flow
   turns all tokens of the sentence so far into mel. The noise is fixed per sentence, so the mel of earlier frames
   does not change between calls. Only the new frames go to HiFiGAN.
4. HiFiGAN re-renders the last OVERLAP mel frames of the previous chunk together with the new ones, keeps the
   excitation source continuous (cache_source) and crossfades the overlap with a Hamming window
   (the CosyVoice 2 method, Apache-2.0). No clicks, no gaps.
5. Between sentences: 0.25 s of silence, as in the whole-text path.

Speed switches (measured in tools/bench_stream.py, see docs/STREAMING.md):
  dtype      T3 in bf16 (RTX 30xx) or fp16 (T4). TF32 matmuls are on for fp32 on Ampere.
  cfm_steps  flow-matching steps in S3Gen (10 = original).
  dec_cond   seconds of the reference clip S3Gen attends to (10 = original). Every flow call pays for all of it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import uz_tts
from live import SlotPool  # noqa: F401  (torch-free, shared with the tests)
from uznorm import for_model, normalize, split_sentences

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
HOP = 480              # audio samples per mel frame in S3Gen
TOKEN_FRAMES = 2       # mel frames per speech token
LOOKAHEAD = 3          # the flow's own look-ahead (tokens)
HOLDBACK = 8           # tokens of real right context every emitted frame must have. The flow attends to the whole
                       # input and was not trained for streaming: frames near the right edge change once more
                       # tokens arrive, so emitting them early leaves seams (UTMOS 3.60 -> 3.47 with 3, 2026-11-15)
OVERLAP = 4            # mel frames re-rendered and crossfaded at each chunk border (80 ms, held back each chunk)
PAUSE = 0.25           # seconds between sentences (same as the whole-text path)
FIRST_CHUNK = 16       # tokens in the first chunk: 0.6 s of speech at once, enough buffer for the next call
CHUNK = 40             # most tokens per later chunk (1.6 s of speech); the Pacer often sends sooner
MIN_CHUNK = 12         # fewest new tokens per later chunk (0.5 s): fewer seams; the Pacer raises it when calls are dear
CONTEXT = 30           # tokens of left context the flow sees besides the new ones (1.2 s)
TOKEN_SEC = 0.04       # seconds of speech per token (25 tokens/s)
MARGIN = 0.08          # seconds of audio we want left in the listener's buffer when the next chunk lands
STOP_SPEECH = 6561     # speech token ids at or above this are special
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def split_text(text: str) -> list[str]:
    """Original sentences (for marks); each is normalised separately later."""
    return split_sentences(text.strip(), max_chars=10_000) if text.strip() else []


class Pacer:
    """How far audio production is ahead of the listener, who starts playing at the first sample of a turn.

    Chunks are cut by need, not by a fixed size. After the first chunk the buffer is short (~0.3 s), so the next
    vocoder call comes after a few tokens. As the buffer grows, chunks grow up to CHUNK tokens (fewer calls, less
    GPU). The same Pacer runs across all sentences of a turn, so a sentence starts while the last one still plays."""

    def __init__(self):
        self.t0 = None
        self.sent = 0.0
        self.cost = 0.15  # seconds per vocoder call, a running average
        self.t3 = 0.015   # seconds per token, a running average

    def emitted(self, seconds: float) -> None:
        if self.t0 is None:
            self.t0 = time.perf_counter()
        self.sent += seconds

    def ahead(self) -> float:
        return float("inf") if self.t0 is None else self.sent - (time.perf_counter() - self.t0)

    def observe(self, call_seconds: float) -> None:
        self.cost = 0.7 * self.cost + 0.3 * call_seconds

    def token(self, seconds: float) -> None:
        self.t3 = 0.9 * self.t3 + 0.1 * seconds

    def min_tokens(self, most: int) -> int:
        """Fewest tokens a chunk needs so the audio it adds pays for the vocoder call that makes it.
        Smaller chunks when behind only make more calls and fall further behind (seen on a T4, 2026-11-15)."""
        gain = TOKEN_SEC - self.t3  # audio gained per token, net of the time spent making it
        if gain <= 0.005:
            return most
        import math
        return max(MIN_CHUNK, min(most, math.ceil(1.2 * self.cost / gain)))


class Streamer:
    def __init__(self, engine, prefix, params: dict, dtype: str = "fp32", cfm_steps: int = 10,
                 first_chunk: int | None = None, chunk: int | None = None, decoders: list | None = None):
        self.engine, self.prefix, self.params = engine, prefix, params
        self.dtype_name, self.dtype = dtype, DTYPES[dtype]
        self.cfm_steps = cfm_steps
        self.first_chunk, self.chunk = first_chunk or FIRST_CHUNK, chunk or CHUNK
        self.sr = SR
        self.device = engine.device
        # One slot per parallel stream. With CUDA graphs each slot owns a GraphDecoder (its own KV cache);
        # without them there is a single slot and transformers' generate loop.
        self.slots = SlotPool(decoders or [None])
        self.streams = len(decoders or [None])
        self.t3_timer = self.s3_timer = None
        self.window = torch.from_numpy(np.hamming(2 * OVERLAP * HOP).astype(np.float32)).to(self.device)
        # Flow noise for the reference-prompt frames, fixed once. With it, noised_mels covers the whole input,
        # so the decoder draws no fresh random numbers: the same tokens always give the same mel.
        prompt = engine.conds.gen["prompt_feat"].shape[1]
        g = torch.Generator(device=self.device).manual_seed(777)
        self.prompt_noise = torch.randn(1, 80, prompt, generator=g, device=self.device)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, profile_name: str = "ovozim", dtype: str = "fp32", cfm_steps: int = 10,
             first_chunk: int | None = None, chunk: int | None = None, dec_cond: float | None = None,
             ref: str | None = None, graphs: bool = False, streams: int = 1, max_len: int = 1400,
             flow_half: bool = False, flow_graphs: bool = False) -> "Streamer":
        import os
        profile = json.loads((ROOT / "data" / "voices" / f"{profile_name}.json").read_text(encoding="utf-8"))
        params = {**uz_tts.PARAMS, **profile.get("params", {})}
        ref = ref or os.environ.get("TILSIM_REF") or profile["ref"]
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True   # Ampere+: fp32 matmuls on tensor cores
            torch.backends.cudnn.allow_tf32 = True
        engine = uz_tts.load_engine()
        engine.ENC_COND_LEN = int(float(profile.get("prompt_sec", 6)) * 16000)
        if dec_cond:
            engine.DEC_COND_LEN = int(dec_cond * SR)
        engine.prepare_conditionals(ref, exaggeration=params["exaggeration"])
        prefix = uz_tts.Prefix(engine, ref, profile["prefix_end"], profile["prefix_text"])
        if dtype != "fp32":
            engine.t3.to(DTYPES[dtype])
            engine.conds.t3.to(dtype=DTYPES[dtype])
        if flow_half and torch.cuda.is_available():  # the 56-block flow UNet: bf16 on RTX 30xx+, fp16 on T4
            half = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            engine.s3gen.flow.decoder.estimator.to(half)
        if flow_graphs and torch.cuda.is_available():
            from fastflow import FlowGraphs
            prompt = engine.conds.gen["prompt_feat"].shape[1]
            # the flow sees prompt + CONTEXT + at most CHUNK new + HOLDBACK tokens: buckets up to that, no more
            longest = prompt + TOKEN_FRAMES * (CONTEXT + max(chunk or CHUNK, first_chunk or FIRST_CHUNK) + HOLDBACK + 8)
            sizes = [b for b in range(256, 1408, 64) if b >= prompt + TOKEN_FRAMES * 8]
            FlowGraphs(engine.s3gen, buckets=[b for b in sizes if b <= longest + 64]).capture()
        decoders = None
        if graphs and torch.cuda.is_available():
            from fastt3 import GraphDecoder
            decoders = []
            for _ in range(max(1, streams)):
                d = GraphDecoder(engine.t3, max_len=max_len)
                d.capture()
                decoders.append(d)
        return cls(engine, prefix, params, dtype, cfm_steps, first_chunk, chunk, decoders)

    def warmup(self) -> None:
        """First CUDA kernels and allocator growth are slow; pay for them before the first listener."""
        for text in ("Salom.", "Assalomu alaykum, qalaysiz?"):
            for _ in self.stream(text):
                pass

    # ------------------------------------------------------------ public
    def stream(self, text: str, cancel: threading.Event | None = None, seed: int = 0, priority: int = 1,
               pacer: "Pacer | None" = None):
        """Yield {"type": "mark"}, {"type": "audio"} events for the whole text; stops quietly on cancel.
        priority: 0 live conversation, 1 API request, 2 website demo, 3 background job (lower goes first)."""
        sentences = split_text(text)
        pacer = pacer or Pacer()
        for i, original in enumerate(sentences):
            if cancel is not None and cancel.is_set():
                return
            yield {"type": "mark", "text": original, "index": i}
            pieces = split_sentences(normalize(original), max_chars=200)
            for k, spoken in enumerate(pieces):
                for pcm in self.sentence(spoken, seed + i * 7 + k, cancel, priority, pacer):
                    yield {"type": "audio", "pcm": pcm}
                if cancel is not None and cancel.is_set():
                    return
                if k < len(pieces) - 1 or i < len(sentences) - 1:
                    pacer.emitted(PAUSE)
                    yield {"type": "audio", "pcm": np.zeros(int(PAUSE * SR), np.float32)}

    def sentence(self, spoken: str, seed: int = 0, cancel: threading.Event | None = None, priority: int = 1,
                 pacer: "Pacer | None" = None, whole: bool = False):
        """One normalised sentence -> float32 audio chunks. Waits for a free slot; lower priority numbers first,
        so a live conversation never waits behind more than one sentence of a background job."""
        slot = self.slots.acquire(priority)
        try:
            yield from self._sentence(spoken, seed, cancel, slot, pacer or Pacer(), whole)
        finally:
            self.slots.release(slot)

    # ------------------------------------------------------------ internals
    def _timed(self, timer, fn, *a, **kw):
        if timer is None:
            return fn(*a, **kw)
        return timer.wrap(fn)(*a, **kw)

    @torch.inference_mode()
    def _sentence(self, spoken: str, seed: int, cancel, slot=None, pacer: "Pacer | None" = None, whole: bool = False):
        """whole=True: one vocoder call for the finished sentence (no seams) for /v1/tts and jobs, where nobody is
        listening yet. Live sessions use chunks."""
        from chatterbox.tts import punc_norm

        eng, p = self.engine, self.params
        rng = torch.Generator(device=self.device).manual_seed(seed)  # per stream: parallel streams do not mix
        text = for_model(spoken)
        cond = eng.conds.t3
        if p["exaggeration"] != float(cond.emotion_adv.view(-1)[0]):
            from chatterbox.models.t3.modules.cond_enc import T3Cond
            eng.conds.t3 = cond = T3Cond(speaker_emb=cond.speaker_emb, cond_prompt_speech_tokens=cond.cond_prompt_speech_tokens,
                                         emotion_adv=p["exaggeration"] * torch.ones(1, 1, 1)).to(device=self.device, dtype=self.dtype)
        tokens = eng.tokenizer.text_to_tokens(punc_norm(f"{self.prefix.text} {text}")).to(self.device)
        tokens = torch.cat([tokens, tokens], dim=0)  # CFG pair
        tokens = F.pad(F.pad(tokens, (1, 0), value=eng.t3.hp.start_text_token), (0, 1), value=eng.t3.hp.stop_text_token)
        letters = sum(ch.isalpha() for ch in text)
        max_new = min(1000, 80 + 5 * letters)
        if slot is not None:  # the static KV cache must hold prompt + speech
            max_new = min(max_new, slot.max_len - 400)
        # fixed flow noise for the whole sentence: earlier frames come out the same in every chunk
        g = torch.Generator(device=self.device).manual_seed(seed + 12345)
        noise = torch.randn(1, 80, TOKEN_FRAMES * (max_new + 1), generator=g, device=self.device)

        pacer = pacer or Pacer()
        state = {"done": 0, "cache": None, "first": True}
        buf: list[int] = []
        emitted = 0  # tokens already turned into audio
        gen = self._tokens(cond, tokens, max_new, slot, rng)

        def vocode(final: bool):
            t = time.perf_counter()
            pcm = self._timed(self.s3_timer, self._vocode, buf, noise, state, final)
            pacer.observe(time.perf_counter() - t)
            if pcm.size:
                pacer.emitted(len(pcm) / SR)
            return pcm

        while True:
            t0 = time.perf_counter()
            tok = next(gen, None)
            dt = time.perf_counter() - t0
            if self.t3_timer is not None:
                self.t3_timer.total += dt
            if tok is None or tok >= STOP_SPEECH:
                break
            pacer.token(dt)
            buf.append(tok)
            if cancel is not None and cancel.is_set():
                gen.close()
                return
            new = len(buf) - HOLDBACK - emitted
            if whole:
                due = False
            elif pacer.t0 is None:   # nothing playing yet: first sound as early as possible
                due = new >= self.first_chunk
            else:                  # playing: cut when the buffer would run dry during the next call
                due = new >= self.chunk or (new >= pacer.min_tokens(self.chunk)
                                            and pacer.ahead() - pacer.cost < MARGIN)
            if due:
                pcm = vocode(False)
                emitted = len(buf) - HOLDBACK
                if pcm.size:
                    yield pcm
        gen.close()
        if buf and not (cancel is not None and cancel.is_set()):
            pcm = vocode(True)
            if pcm.size:
                yield pcm

    def _tokens(self, cond, text_tokens: torch.Tensor, max_new: int, dec=None, rng=None):
        """uz_tts._continue_tokens as a generator: yields each new speech token id as soon as it is sampled.
        With a GraphDecoder (engine/fastt3.py) every step is one CUDA-graph replay instead of transformers."""
        from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor
        from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

        t3, p = self.engine.t3, self.params
        cfg_weight, temperature = p["cfg_weight"], p["temperature"]
        with torch.inference_mode():
            initial = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            embeds, _ = t3.prepare_input_embeds(t3_cond=cond, text_tokens=text_tokens, speech_tokens=initial,
                                                cfg_weight=cfg_weight)
            model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb,
                                         speech_head=t3.speech_head, alignment_stream_analyzer=None)
            dev = embeds.device
            bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=dev)
            prefix = self.prefix.tokens
            n_pre = prefix.numel()
            head = torch.cat([t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0),
                              t3.speech_emb(prefix.view(1, -1)) +
                              t3.speech_pos_emb.get_fixed_embedding(torch.arange(1, n_pre + 1, device=dev))], dim=1)
            full = torch.cat([embeds, torch.cat([head, head])], dim=1)
            past = None
            if dec is not None:
                step = dec.prefill(full).float()
            else:
                out = model(inputs_embeds=full, past_key_values=None, use_cache=True, output_attentions=False,
                            output_hidden_states=True, return_dict=True)
                past, step = out.past_key_values, out.logits[:, -1, :].float()
            rep = RepetitionPenaltyLogitsProcessor(penalty=float(p["repetition_penalty"]))
            min_p_w = MinPLogitsWarper(min_p=0.05)
            generated = bos.clone()
            for i in range(max_new):
                logits = step[0:1] + cfg_weight * (step[0:1] - step[1:2])
                logits = rep(generated, logits)
                if temperature != 1.0:
                    logits = logits / temperature
                logits = min_p_w(generated, logits)  # top_p is 1.0 (keeps everything): its sort is skipped
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=rng)
                tok = int(nxt)
                yield tok
                if tok == t3.hp.stop_speech_token:
                    return
                generated = torch.cat([generated, nxt], dim=1)
                emb = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(n_pre + i + 1)
                emb = torch.cat([emb, emb])
                if dec is not None:
                    step = dec.step(emb).float()
                else:
                    out = model(inputs_embeds=emb, past_key_values=past, output_attentions=False,
                                output_hidden_states=True, return_dict=True)
                    past, step = out.past_key_values, out.logits[:, -1, :].float()

    @torch.inference_mode()
    def _vocode(self, buf: list[int], noise: torch.Tensor, state: dict, finalize: bool) -> np.ndarray:
        s3 = self.engine.s3gen
        ref = self.engine.conds.gen
        n_all = len(buf)
        usable = TOKEN_FRAMES * n_all - (0 if finalize else TOKEN_FRAMES * HOLDBACK)
        if usable <= state["done"] and not finalize:
            return np.zeros(0, np.float32)
        # Only the last CONTEXT tokens before the new ones go to the flow: its cost stays flat however long the
        # sentence is (without this a 15 s sentence made every chunk ~4x dearer and long texts fell behind).
        ctx0 = max(0, state["done"] // TOKEN_FRAMES - CONTEXT)
        toks = torch.tensor([buf[ctx0:]], dtype=torch.long, device=self.device)
        n = toks.shape[1]
        # Always finalize=True on the tokens given, then keep only frames whose look-ahead tokens exist.
        # (chatterbox 0.1.7's finalize=False path cuts the mel but not its mask and crashes.)
        mel, _ = s3.flow.inference(
            token=toks, token_len=torch.tensor([n], device=self.device),
            prompt_token=ref["prompt_token"], prompt_token_len=ref["prompt_token_len"],
            prompt_feat=ref["prompt_feat"], prompt_feat_len=ref.get("prompt_feat_len"),
            embedding=ref["embedding"], finalize=True, n_timesteps=self.cfm_steps,
            noised_mels=torch.cat([self.prompt_noise, noise[:, :, TOKEN_FRAMES * ctx0:TOKEN_FRAMES * n_all]], dim=2),
            meanflow=getattr(s3, "meanflow", False))
        off = TOKEN_FRAMES * ctx0  # mel frame 0 here is absolute frame `off`
        new = mel[:, :, state["done"] - off:usable - off].to(s3.dtype)
        state["done"] = usable
        cache = state["cache"]
        if cache is not None:
            new = torch.cat([cache["mel"], new], dim=2)
            src = cache["source"]
        else:
            src = torch.zeros(1, 1, 0, device=self.device, dtype=s3.dtype)
        if new.shape[2] == 0:
            return np.zeros(0, np.float32)
        wav, source = s3.mel2wav.inference(speech_feat=new, cache_source=src)
        wav = wav[0]
        ov = OVERLAP * HOP
        if cache is not None:  # crossfade the re-rendered overlap with the tail we held back last time
            k = min(ov, wav.shape[0], cache["speech"].shape[0])
            wav = wav.clone()
            wav[:k] = wav[:k] * self.window[:k] + cache["speech"][-k:] * self.window[ov:ov + k]
        if state["first"]:  # same ad-hoc fade as S3Gen.inference: no spill-over from the reference clip
            fade = s3.trim_fade.to(wav.dtype)
            k = min(len(fade), wav.shape[0])
            wav[:k] = wav[:k] * fade[:k]
            state["first"] = False
        if not finalize and new.shape[2] > OVERLAP and wav.shape[0] > ov:
            state["cache"] = {"mel": new[:, :, -OVERLAP:], "source": source[:, :, -ov:], "speech": wav[-ov:].clone()}
            wav = wav[:-ov]
        elif not finalize:
            state["cache"] = None
        out = wav.float().cpu().numpy()
        if state.get("head", True):  # first audio of the sentence: drop leading silence (as uz_tts.trim does)
            out = head_trim(out)
            state["head"] = False
        if finalize:
            out = tail_trim(out)
        return out


def head_trim(x: np.ndarray, sr: int = SR, head_pad: float = 0.05) -> np.ndarray:
    hop = int(sr * 0.01)
    n = len(x) // hop
    if n < 3:
        return x
    db = 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    audible = np.where(db > max(db.max() - 45, -60))[0]
    if not len(audible):
        return x
    start = max(0, audible[0] * hop - int(head_pad * sr))
    y = x[start:].copy()
    k = min(int(0.008 * sr), len(y) // 2)
    if k:
        y[:k] *= np.linspace(0, 1, k)
    return y


def tail_trim(x: np.ndarray, sr: int = SR, tail_pad: float = 0.15) -> np.ndarray:
    """The end half of uz_tts.trim for the last chunk of a sentence: drop sound after a >=60 ms gap
    (breaths, gasps), keep the attached ending, fade out over 40 ms."""
    hop = int(sr * 0.01)
    n = (len(x) - hop) // hop
    if n <= 4:
        return x
    frames = x[:n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    peak = db.max()
    audible = db > max(peak - 45, -60)
    strong = np.where((db > peak - 30) & (zcr < 0.25))[0]
    if not len(strong):
        return x
    last = int(strong[-1])
    gap, i = 0, last
    while i + 1 < n and i - last < 25:
        i += 1
        if audible[i]:
            gap = 0
        else:
            gap += 1
            if gap >= 6:
                break
    end = min(len(x), (i - gap + 1) * hop + int(tail_pad * sr))
    y = x[:end].copy()
    k = min(int(0.04 * sr), len(y) // 2)
    if k:
        y[-k:] *= np.linspace(1, 0, k) ** 2
    return y
