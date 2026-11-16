"""Tilsim voice API: Uzbek text -> speech in the owner's voice, on this laptop's GPU.

    E:/tools/voiceclone/.venv/Scripts/python.exe engine/serve.py      # http://127.0.0.1:8790 (start-tilsim.bat adds the tunnel)

Two kinds of callers:
  - the website's text box: no key, browser Origin must be the Tilsim site, small public limits;
  - your own apps with an API key (engine/apikeys.py): bigger texts, OGG/WAV, jobs for long texts, usage numbers.

Endpoints (full guide with examples: docs/API.md)
    GET  /v1/health                 public: is the model loaded, queue length
    GET  /v1/voices                 the voices you can use
    POST /v1/tts                    {"text", "voice"?, "format"?: mp3|wav|ogg, "seed"?} -> audio bytes (short texts)
    POST /v1/normalize              {"text"} -> how Tilsim will read it (numbers as words, sentences); no GPU
    POST /v1/jobs                   same body as /v1/tts, for long texts -> 202 {"id", ...}
    GET  /v1/jobs/{id}              status + progress (sentences done / total)
    GET  /v1/jobs/{id}/audio        the finished audio
    GET  /v1/usage                  this key's requests, characters and audio seconds this month

One GPU worker makes one sentence at a time (4 GB card). Requests wait in a priority queue:
API keys first, then the website, then background jobs. A sentence whose speed is far off for its length
(cut short or runaway) is made again with another seed, up to 3 tries.
Texts are never written to disk. The log keeps time, key name or a hashed visitor id, length and duration.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import itertools
import json
import queue
import secrets
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

from apikeys import DEFAULT_PATH as KEYS_PATH  # noqa: E402
from apikeys import KeyStore  # noqa: E402
from uznorm import normalize, split_sentences  # noqa: E402

# Settings the cloud copy changes through the environment (deploy/modal_app.py)
KEYS_FILE = Path(os.environ.get("TILSIM_KEYS", KEYS_PATH))
REF_OVERRIDE = os.environ.get("TILSIM_REF")                    # the voice clip's path inside the container
DEMO_DAILY_MAX = int(os.environ.get("TILSIM_DEMO_DAILY_MAX", "0")) or None   # all site demos per day (cloud credits)
# Engine: "stream" = engine/stream.py (half precision + CUDA graphs, live sessions, ~3 GB VRAM; serves /v1/tts too),
# "classic" = the fp32 whole-text path of engine/uz_tts.py (no live streaming),
# "both" = classic for /v1/tts and jobs (best CER) + stream for live sessions (~7 GB: the cloud T4).
ENGINE = os.environ.get("TILSIM_ENGINE", "stream")
STREAMS = int(os.environ.get("TILSIM_STREAMS", "1"))          # parallel live streams (each its own KV cache)
CFM_STEPS = int(os.environ.get("TILSIM_CFM_STEPS", "6"))      # flow steps in the vocoder (10 = original; 6 scored the same)
T3_DTYPE = os.environ.get("TILSIM_T3_DTYPE", "")             # token model precision: fp32 on the T4 (CER 1.53 -> 1.37%,
                                                             # fixes «o'n ikkinchi»); bf16 on the 4 GB laptop (VRAM)
DEC_COND = float(os.environ.get("TILSIM_DEC_COND", "10"))    # seconds of the voice clip the vocoder attends to
                                                             # (10 = original; 4 and 6 cost CER, docs/STREAMING.md)
STREAM_MAX_CHARS = 2000                                       # /v1/tts/stream body
MAX_SESSIONS_PER_STREAM = 2                                   # open WebSockets per GPU stream slot
TOKEN_MAX_TTL = 3600                                          # browser tokens live at most an hour
# website (no key)
DEMO_MAX_CHARS = 300
DEMO_PER_IP = [(600, 8), (86400, 60)]   # (window seconds, max requests) per visitor
DEMO_WAITING_MAX = 4
ORIGINS = {"https://javohir120.github.io", "http://localhost:8780", "http://127.0.0.1:8780"}
# API keys
SYNC_MAX_CHARS = 600      # /v1/tts must answer within the tunnel's 100 s; longer texts go to /v1/jobs
MAX_SPOKEN_RATIO = 2.5    # numbers grow when written as words; cap the spoken text at this x the limit
ACTIVE_JOBS_PER_KEY = 3
JOB_KEEP_SECONDS = 3600
# shared
QUEUE_MAX = 12
CACHE_SIZE = 32         # raw float audio: ~1 MB per 10 s clip
PRIORITY = {"key": 0, "demo": 1, "job": 2}
SLOT_PRIORITY = {"live": 0, "key": 1, "demo": 2, "job": 3}   # GPU slots: a live conversation goes first
FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}
VOICES = {"asoschi": {"profile": "ovozim", "name": "Asoschi", "language": "uz", "gender": "male",
                      "description": "Loyiha asoschisining ovozi: tinch, aniq, iliq."}}
DEFAULT_VOICE = "asoschi"
LOG = ROOT / "out" / "serve.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_priority_above_normal() -> None:
    """The owner wants voicelab jobs at AboveNormal (fast, and the desktop still responds)."""
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo handle; the int default truncates it
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000):
            log("could not raise process priority")


def encode(audio, sr: int, fmt: str) -> bytes:
    """float32 mono -> MP3 / WAV / OGG-Opus, all at -16 LUFS (the site's demo clips use the same loudness).
    OGG-Opus mono is what Telegram's sendVoice wants."""
    codec = {"mp3": ["-ar", str(sr), "-codec:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"],
             "wav": ["-ar", str(sr), "-codec:a", "pcm_s16le", "-f", "wav"],
             "ogg": ["-ar", "48000", "-codec:a", "libopus", "-b:a", "48k", "-f", "ogg"]}[fmt]
    # loudnorm crashes ffmpeg on pure digital silence (seen 2026-11-15, exit 0xC0000409), so skip it there
    level = ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if audio.size and float(abs(audio).max()) > 1e-4 else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-f", "f32le", "-ar", str(sr), "-ac", "1",
           "-i", "-", *level, *codec, "-"]
    return subprocess.run(cmd, input=audio.astype("float32").tobytes(), capture_output=True, check=True).stdout


def _b64(b: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(rec: dict, ttl: int) -> tuple[str, int]:
    """A short-lived browser token for one key: tlst_<key id + expiry>.<HMAC>.
    Signed with the key's stored hash: only this server (or the key's owner) can make one, and revoking the
    key kills its tokens. The real key never reaches the student's browser."""
    import hmac
    exp = int(time.time()) + ttl
    payload = f"{rec['id']}.{exp}"
    sig = hmac.new(rec["hash"].encode(), payload.encode(), hashlib.sha256).digest()[:24]
    return "tlst_" + _b64(payload.encode()) + "." + _b64(sig), exp


def check_token(keys: KeyStore, token: str) -> dict | None:
    import hmac
    if not token or not token.startswith("tlst_"):
        return None
    try:
        p64, s64 = token[5:].split(".")
        payload = _unb64(p64).decode()
        key_id, exp = payload.rsplit(".", 1)
        if int(exp) < time.time():
            return None
        sig = _unb64(s64)
    except (ValueError, UnicodeDecodeError):
        return None
    rec = keys.active_by_id(key_id)
    if rec is None:
        return None
    want = hmac.new(rec["hash"].encode(), payload.encode(), hashlib.sha256).digest()[:24]
    return rec if hmac.compare_digest(want, sig) else None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


# Optional voice style of a request or a live "text" message (2026-11-15). Left out = the old behaviour exactly.
LANGS = ("uz", "en", "auto")
EMOTIONS = {"neytral": "neutral", "iliq": "warm", "quvnoq": "cheerful", "hayajonli": "excited",
            "xotirjam": "calm", "jiddiy": "serious", "hamdard": "sympathetic", "hayron": "surprised"}


def parse_style(body: dict, english: bool, emotions: bool = True) -> dict:
    """lang / emotion / intensity -> a dict holding only the fields given. Raises ApiError on a bad value.
    emotions=False: the voice has no emotion bank yet, so any emotion but "neytral" is refused, not silently ignored."""
    out: dict = {}
    if body.get("lang") is not None:
        lang = str(body["lang"]).lower()
        if lang not in LANGS:
            raise ApiError(400, "bad_lang", "lang: uz, en yoki auto.")
        if lang == "en" and not english:
            raise ApiError(400, "lang_unavailable", "Bu serverda inglizcha rejim yoqilmagan.")
        out["lang"] = lang
    if body.get("emotion") is not None:
        emotion = str(body["emotion"]).lower()
        if emotion not in EMOTIONS:
            raise ApiError(400, "bad_emotion", "emotion: " + ", ".join(EMOTIONS) + ".")
        if emotion != "neytral" and not emotions:
            raise ApiError(400, "emotion_unavailable", "Emotsiyalar hali tayyor emas: hozircha faqat \"neytral\". "
                                                      "Tayyor bo‘lganda /v1/voices ro‘yxatida ko‘rinadi.")
        out["emotion"] = emotion
    if body.get("intensity") is not None:
        try:
            x = float(body["intensity"])
        except (TypeError, ValueError):
            x = -1.0
        if not 0.0 <= x <= 1.0:   # also rejects NaN
            raise ApiError(400, "bad_intensity", "intensity 0 dan 1 gacha son bo‘lsin.")
        out["intensity"] = x
    return out


def is_plain(style: dict) -> bool:
    """Uzbek, neutral: the request takes exactly the path it took before styles existed."""
    return style.get("lang", "uz") == "uz" and style.get("emotion", "neytral") == "neytral"


class Voice:
    """The loaded model plus one voice profile; used by the single worker thread only."""

    def __init__(self, profile_name: str):
        import numpy as np
        import torch
        import uz_tts

        self.np, self.torch, self.uz = np, torch, uz_tts
        profile = json.loads((ROOT / "data" / "voices" / f"{profile_name}.json").read_text(encoding="utf-8"))
        self.params = {**uz_tts.PARAMS, **profile.get("params", {})}
        ref = REF_OVERRIDE or profile["ref"]
        t0 = time.time()
        self.engine = uz_tts.load_engine()
        self.sr = self.engine.sr
        self.engine.ENC_COND_LEN = int(float(profile.get("prompt_sec", 6)) * 16000)
        self.engine.prepare_conditionals(ref, exaggeration=self.params["exaggeration"])
        self.prefix = uz_tts.Prefix(self.engine, ref, profile["prefix_end"], profile["prefix_text"])
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        log(f"model ready on {self.engine.device} in {time.time() - t0:.0f} s, VRAM {vram:.2f} GB")

    def sentence(self, text: str, seed: int):
        """One spoken sentence (already normalised). Retake if its speed is implausible for its length."""
        from uznorm import for_model
        letters = max(1, sum(ch.isalpha() for ch in text))
        best, best_gap = None, 1e9
        for attempt in range(3):
            self.torch.manual_seed(seed + 1000 * attempt)  # attempt 0 = the same seed engine/uz_tts.speak uses
            wav = self.uz.generate_continuing(self.engine, self.prefix, for_model(text), **self.params)
            wav = self.uz.trim(wav, self.sr)
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()  # 4 GB card: keep cached blocks from piling up
            rate = letters / max(len(wav) / self.sr, 1e-3)  # the owner speaks ~12-14 letters/s
            if 7 <= rate <= 22:
                return wav
            if abs(rate - 13) < best_gap:
                best, best_gap = wav, abs(rate - 13)
            log(f"retake: {rate:.1f} letters/s on try {attempt + 1}")
        return best

    def synth(self, sentences: list[str], seed: int = 0, progress=None):
        pause = self.np.zeros(int(0.25 * self.sr), self.np.float32)
        pieces = []
        for i, s in enumerate(sentences):
            pieces += [self.sentence(s, seed + i), pause]
            if progress:
                progress(i + 1)
        return self.np.concatenate(pieces[:-1]) if pieces else self.np.zeros(0, self.np.float32)


class StreamVoice:
    """/v1/tts and /v1/jobs on the streaming engine, so one model in VRAM serves everything.
    The whole sentence is made before it is used, so the speed check and retake of Voice still apply."""

    def __init__(self, streamer):
        import numpy as np
        self.s, self.sr, self.np = streamer, streamer.sr, np

    def sentence(self, spoken: str, seed: int, priority: int = 1, style=None):
        np = self.np
        letters = max(1, sum(ch.isalpha() for ch in spoken))
        best, best_gap = None, 1e9
        for attempt in range(3):
            parts = list(self.s.sentence(spoken, seed + 1000 * attempt, None, priority, whole=True, style=style))
            wav = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            rate = letters / max(len(wav) / self.sr, 1e-3)
            if 7 <= rate <= 22:
                return wav
            if abs(rate - 13) < best_gap:
                best, best_gap = wav, abs(rate - 13)
            log(f"retake: {rate:.1f} letters/s on try {attempt + 1}")
        return best

    def synth(self, sentences: list[str], seed: int = 0, progress=None, priority: int = 1):
        np = self.np
        pause = np.zeros(int(0.25 * self.sr), np.float32)
        pieces = []
        for i, s in enumerate(sentences):
            pieces += [self.sentence(s, seed + i, priority), pause]
            if progress:
                progress(i + 1)
        return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)

    def synth_plan(self, plan: list[tuple[str, str]], seed: int, style: dict, progress=None, priority: int = 1):
        """[(lang, spoken sentence)] in one emotion: the styled /v1/tts and jobs."""
        np = self.np
        pause = np.zeros(int(0.25 * self.sr), np.float32)
        pieces = []
        for i, (lang, s) in enumerate(plan):
            st = self.s.style(lang, style.get("emotion"), style.get("intensity"))
            pieces += [self.sentence(s, seed + i, priority, st), pause]
            if progress:
                progress(i + 1)
        return np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)


def load_streamer(streams: int = STREAMS):
    import torch

    import stream
    half = "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    fast = torch.cuda.is_available()
    s = stream.Streamer.load(VOICES[DEFAULT_VOICE]["profile"], dtype=(T3_DTYPE or half) if fast else "fp32", cfm_steps=CFM_STEPS,
                             dec_cond=DEC_COND, ref=REF_OVERRIDE, graphs=fast, streams=streams, flow_half=fast,
                             flow_graphs=fast)
    s.warmup()
    return s


class Service:
    def __init__(self, keys: KeyStore, preload: bool = False):
        """preload: load the model before returning (the cloud opens its port only when the voice is ready);
        otherwise it loads in the background and requests meanwhile get 503 "loading"."""
        self.keys = keys
        self.voice: Voice | None = None
        self.jobs_q: queue.PriorityQueue = queue.PriorityQueue(maxsize=QUEUE_MAX)
        self.seq = itertools.count()
        self.busy = 0
        self.demo_waiting = 0
        self.demo_day: tuple[str, int] = ("", 0)
        self.lock = threading.Lock()
        self.ip_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.jobs: dict[str, dict] = {}
        self.streamer = None          # engine/stream.py Streamer when ENGINE is "stream" or "both"
        self.sessions = 0             # open live (WebSocket / HTTP stream) sessions
        if preload:
            self._load()
            threading.Thread(target=self._loop, daemon=True).start()
        else:
            threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- GPU worker
    def _load(self) -> None:
        t0 = time.time()
        if ENGINE in ("classic", "both"):  # /v1/tts and jobs exactly as before: fp32, whole sentences, retakes
            self.voice = Voice(VOICES[DEFAULT_VOICE]["profile"])
            self.voice.synth(["Salom."])  # warm-up: first CUDA kernels are slow
        if ENGINE in ("stream", "both"):   # live streaming (half precision + CUDA graphs)
            self.streamer = load_streamer()
            if self.voice is None:         # one engine for everything (the 4 GB laptop)
                self.voice = StreamVoice(self.streamer)
        log(f"{ENGINE} engine ready in {time.time() - t0:.0f} s, accepting requests")

    def _worker(self) -> None:
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            log(f"model failed to load: {e!r}")
            return
        self._loop()

    def _loop(self) -> None:
        while True:
            _, _, kind, payload, fut = self.jobs_q.get()
            if kind == "demo":
                with self.lock:
                    self.demo_waiting -= 1
            self.busy = 1
            try:
                if kind == "job":
                    self._run_job(payload)
                else:
                    fut.set_result(self._make(payload, priority=SLOT_PRIORITY[kind]))
            except Exception as e:  # noqa: BLE001
                if fut:
                    fut.set_exception(e)
            finally:
                self.busy = 0

    def _make(self, req: dict, progress=None, priority: int = 1) -> dict:
        """GPU work only: raw float audio. Encoding happens on the caller's thread, so the GPU never waits for ffmpeg."""
        if req.get("plan") is not None:   # English / auto / emotion: always the streaming engine
            sv = self.voice if isinstance(self.voice, StreamVoice) else StreamVoice(self.streamer)
            audio = sv.synth_plan(req["plan"], req["seed"], req["style"], progress, priority=priority)
            return {"spoken": req["spoken"], "pcm": audio, "sr": sv.sr, "seconds": round(len(audio) / sv.sr, 2)}
        if isinstance(self.voice, StreamVoice):
            audio = self.voice.synth(req["sentences"], req["seed"], progress, priority=priority)
        else:
            audio = self.voice.synth(req["sentences"], req["seed"], progress)
        return {"spoken": req["spoken"], "pcm": audio, "sr": self.voice.sr, "seconds": round(len(audio) / self.voice.sr, 2)}

    def _run_job(self, job: dict) -> None:
        job["status"], job["started"] = "working", time.time()

        def progress(done):
            job["done"] = done
        try:
            raw = self._make(job["req"], progress, priority=SLOT_PRIORITY["job"])
            result = {"audio": encode(raw["pcm"], raw["sr"], job["req"]["format"]), "seconds": raw["seconds"]}
            job.update(status="done", result=result, seconds=result["seconds"], finished=time.time())
            self.keys.record(job["key"], len(job["req"]["text"]), result["seconds"])
            log(f"job {job['id']} ({job['key']['name']}) {len(job['req']['text'])} chars -> "
                f"{result['seconds']:.1f} s audio in {job['finished'] - job['started']:.0f} s")
        except Exception as e:  # noqa: BLE001
            job.update(status="failed", error="Ovoz yaratishda xato bo‘ldi.", finished=time.time())
            log(f"job {job['id']} failed: {e!r}")

    # ---------------------------------------------------------------- helpers
    def prepare(self, body: dict, max_chars: int) -> dict:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "empty_text", "Matn yozing (\"text\").")
        if len(text) > max_chars:
            raise ApiError(400, "too_long", f"Matn {max_chars} belgidan oshmasin.")
        voice = str(body.get("voice") or DEFAULT_VOICE).lower()
        if voice not in VOICES:
            raise ApiError(400, "unknown_voice", f"Bunday ovoz yo‘q: {voice}. Ro‘yxat: /v1/voices")
        fmt = str(body.get("format") or "mp3").lower()
        if fmt not in FORMATS:
            raise ApiError(400, "bad_format", "format: mp3, wav yoki ogg.")
        try:
            seed = int(body.get("seed", 0))
        except (TypeError, ValueError):
            raise ApiError(400, "bad_seed", "seed butun son bo‘lsin.") from None
        style = parse_style(body, self.english(), self.emotions())
        if not is_plain(style):
            return self._prepare_styled(text, voice, fmt, seed, style, max_chars)
        spoken = normalize(text)
        if len(spoken) > max_chars * MAX_SPOKEN_RATIO:
            raise ApiError(400, "too_long", "Matn so‘zga aylantirilganda juda uzun bo‘ldi (raqamlar ko‘p). Qisqartiring.")
        return {"text": text, "voice": voice, "format": fmt, "seed": seed, "spoken": spoken,
                "sentences": split_sentences(spoken, max_chars=200), "style": style}

    def english(self) -> bool:
        return getattr(self.streamer, "english", None) is not None

    def emotions(self) -> bool:
        """True once the voice profile has an emotion bank (profile["emotions"]); until then only "neytral"."""
        return bool(getattr(self.streamer, "bank", None))

    def _prepare_styled(self, text: str, voice: str, fmt: str, seed: int, style: dict, max_chars: int) -> dict:
        """English, auto or an emotion: made on the streaming engine, sentence by sentence in its language."""
        from uznorm.en_normalize import normalize_en
        if self.streamer is None:
            raise ApiError(400, "style_unavailable", "Bu serverda til va emotsiya tanlash yoqilmagan.")
        plan = []
        for lang, original in self.streamer.plan(text, style.get("lang", "uz")):
            said = normalize_en(original) if lang == "en" else normalize(original)
            plan += [(lang, piece) for piece in split_sentences(said, max_chars=200)]
        spoken = " ".join(piece for _, piece in plan)
        if len(spoken) > max_chars * MAX_SPOKEN_RATIO:
            raise ApiError(400, "too_long", "Matn so‘zga aylantirilganda juda uzun bo‘ldi (raqamlar ko‘p). Qisqartiring.")
        return {"text": text, "voice": voice, "format": fmt, "seed": seed, "spoken": spoken,
                "sentences": [piece for _, piece in plan], "plan": plan, "style": style}

    def demo_wait(self, who: str) -> int:
        now = time.time()
        with self.lock:
            if DEMO_DAILY_MAX:  # all site visitors together, per UTC day: protects the cloud's monthly credit
                day = time.strftime("%Y-%m-%d", time.gmtime(now))
                count = self.demo_day[1] if self.demo_day[0] == day else 0
                if count >= DEMO_DAILY_MAX:
                    return -1
                self.demo_day = (day, count + 1)
            q = self.ip_hits[who]
            while q and now - q[0] > DEMO_PER_IP[-1][0]:
                q.popleft()
            for window, limit in DEMO_PER_IP:
                recent = [t for t in q if now - t <= window]
                if len(recent) >= limit:
                    return int(window - (now - recent[0])) + 1
            q.append(now)
            return 0

    def submit(self, kind: str, payload, fut: Future | None) -> None:
        with self.lock:
            if kind == "demo" and self.demo_waiting >= DEMO_WAITING_MAX:
                raise ApiError(503, "busy", "Hozir navbat to‘la. Birozdan keyin urinib ko‘ring.", 20)
            try:
                self.jobs_q.put_nowait((PRIORITY[kind], next(self.seq), kind, payload, fut))
            except queue.Full:
                raise ApiError(503, "busy", "Hozir navbat to‘la. Birozdan keyin urinib ko‘ring.", 20) from None
            if kind == "demo":
                self.demo_waiting += 1

    def cached(self, req: dict):
        """Raw audio is cached per text+voice+seed, so asking again in another format needs no GPU work."""
        parts = [req["text"], req["voice"], req["seed"]] + ([req["style"]] if req.get("style") else [])
        key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
        return key, self.cache.get(key)

    def remember(self, key: str, result: dict) -> None:
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def sweep_jobs(self) -> None:
        now = time.time()
        for jid in [j for j, v in self.jobs.items() if v.get("finished") and now - v["finished"] > JOB_KEEP_SECONDS]:
            self.jobs.pop(jid, None)


class Handler(BaseHTTPRequestHandler):
    service: Service
    server_version = "Tilsim/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # our own log lines only
        pass

    # ---------------------------------------------------------------- plumbing
    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers",
                             "X-Tilsim-Seconds, X-Tilsim-Spoken, X-Tilsim-Chars, Retry-After")

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: dict, extra: dict | None = None) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _error(self, e: ApiError) -> None:
        self._json(e.status, {"error": e.message, "code": e.code},
                   {"Retry-After": str(e.retry_after)} if e.retry_after else None)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64_000:
            raise ApiError(413, "too_large", "So‘rov juda katta.")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "bad_json", "JSON noto‘g‘ri.") from None
        if not isinstance(body, dict):
            raise ApiError(400, "bad_json", "JSON obyekt yuboring: {\"text\": \"...\"}.")
        return body

    def _caller(self) -> tuple[str, dict | None]:
        """("key", record) for a valid API key, ("demo", None) for the website. Anything else is refused."""
        raw = self.headers.get("X-API-Key") or ""
        auth = self.headers.get("Authorization") or ""
        if not raw and auth.lower().startswith("bearer "):
            raw = auth[7:].strip()
        if raw:
            rec = self.service.keys.find(raw)
            if rec is None:
                raise ApiError(401, "bad_key", "API kalit noto‘g‘ri yoki o‘chirilgan.")
            return "key", rec
        if self.headers.get("Origin", "") in ORIGINS:
            return "demo", None
        raise ApiError(401, "no_key", "API kalit kerak: X-API-Key sarlavhasida yuboring.")

    def _visitor(self) -> str:
        # Cloudflare tunnel sets CF-Connecting-IP; a cloud proxy appends the client to X-Forwarded-For
        # (the last entry is the one the proxy added, so a visitor cannot fake it)
        forwarded = [p.strip() for p in (self.headers.get("X-Forwarded-For") or "").split(",") if p.strip()]
        ip = self.headers.get("CF-Connecting-IP") or (forwarded[-1] if forwarded else self.client_address[0])
        return hashlib.sha256(f"tilsim:{ip}".encode()).hexdigest()[:10]

    def _ready(self) -> None:
        if self.service.voice is None:
            raise ApiError(503, "loading", "Ovoz modeli yuklanmoqda, bir daqiqadan keyin urinib ko‘ring.", 30)

    def _stream_caller(self) -> dict:
        """Live streaming needs a key: X-API-Key / Bearer header, or (browsers cannot set WebSocket headers)
        ?token=tlst_... from /v1/stream/token, or ?key=tlsm_.... Never logged."""
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        token = (q.get("token") or [""])[0]
        if token:
            rec = check_token(self.service.keys, token)
            if rec is None:
                raise ApiError(401, "bad_token", "Token noto‘g‘ri yoki muddati o‘tgan.")
            return rec
        raw = (q.get("key") or [""])[0]
        if raw:
            rec = self.service.keys.find(raw)
            if rec is None:
                raise ApiError(401, "bad_key", "API kalit noto‘g‘ri yoki o‘chirilgan.")
            return rec
        kind, rec = self._caller()
        if kind != "key":
            raise ApiError(401, "no_key", "Jonli ovoz faqat API kalit yoki token bilan ishlaydi.")
        return rec

    def _open_session(self, rec: dict) -> None:
        s = self.service
        self._ready()
        if s.streamer is None:
            raise ApiError(503, "no_stream", "Bu serverda jonli ovoz o‘chirilgan (TILSIM_ENGINE=classic).")
        wait = s.keys.rate_wait(rec)
        if wait:
            raise ApiError(429, "rate_limited", f"Daqiqasiga {rec['per_minute']} so‘rovdan oshdi. {wait} s kuting.", wait)
        with s.lock:
            if s.sessions >= MAX_SESSIONS_PER_STREAM * s.streamer.streams:
                raise ApiError(503, "busy", "Hozir jonli ovoz band. Birozdan keyin urinib ko‘ring.", 10)
            s.sessions += 1

    def _close_session(self) -> None:
        with self.service.lock:
            self.service.sessions -= 1

    def _websocket(self) -> None:
        """GET /v1/stream with Upgrade: websocket. Protocol: docs/API.md, «Jonli ovoz»."""
        from live import LiveSession
        from wsproto import Closed, WebSocket, accept_key

        s = self.service
        if (self.headers.get("Upgrade") or "").lower() != "websocket" or not self.headers.get("Sec-WebSocket-Key"):
            raise ApiError(426, "upgrade_required", "Bu manzil WebSocket orqali ishlaydi (wss://.../v1/stream).")
        rec = self._stream_caller()
        self._open_session(rec)
        try:
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key(self.headers["Sec-WebSocket-Key"]))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            ws = WebSocket(self.rfile, self.wfile)

            def send_event(ev: dict) -> None:
                try:
                    ws.send_text(json.dumps(ev, ensure_ascii=False))
                except Closed:
                    pass

            def send_audio(data: bytes) -> None:
                try:
                    ws.send_bytes(data)
                except Closed:
                    pass

            def usage(chars: int, secs: float) -> None:
                s.keys.record(rec, chars, secs)

            session = LiveSession(s.streamer, send_event, send_audio, max_chars=s.keys.quota_left(rec),
                                  on_usage=usage, priority=SLOT_PRIORITY["live"])
            log(f"live {rec['name']}: connected")
            send_event({"type": "ready", "sample_rate": 24000, "format": "pcm_s16le", "channels": 1})
            try:
                while True:
                    kind, data = ws.recv()
                    if kind != "text":
                        send_event({"type": "error", "code": "bad_message", "message": "JSON matn yuboring."})
                        continue
                    try:
                        msg = json.loads(data)
                        mtype = msg.get("type")
                    except (ValueError, AttributeError):
                        send_event({"type": "error", "code": "bad_json"})
                        continue
                    if mtype == "text":
                        try:
                            style = parse_style(msg, s.english(), s.emotions())
                        except ApiError as e:   # a bad field skips this message; the session stays open
                            send_event({"type": "error", "code": e.code, "message": e.message})
                            continue
                        session.feed(str(msg.get("text", "")), flush=bool(msg.get("flush")), style=style)
                    elif mtype == "flush":
                        session.flush()
                    elif mtype == "cancel":
                        session.cancel()
                    elif mtype == "ping":
                        send_event({"type": "pong"})
                    else:
                        send_event({"type": "error", "code": "bad_type", "message": "type: text | flush | cancel"})
            except Closed:
                pass
            finally:
                session.close()
                ws.close()
                log(f"live {rec['name']}: closed after {session.seconds:.1f} s of audio, {session.chars} chars")
        finally:
            self._close_session()

    def _tts_stream(self, body: dict) -> None:
        """POST /v1/tts/stream: chunked PCM16 / 24 kHz / mono as it is made. For clients without WebSocket."""
        import threading

        from live import pcm16
        s = self.service
        rec = self._stream_caller()
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "empty_text", "Matn yozing (\"text\").")
        if len(text) > STREAM_MAX_CHARS:
            raise ApiError(400, "too_long", f"Matn {STREAM_MAX_CHARS} belgidan oshmasin.")
        left = s.keys.quota_left(rec)
        if left is not None and len(text) > left:
            raise ApiError(402, "quota_exceeded", f"Oylik limit tugadi: {left} belgi qoldi.")
        style = parse_style(body, s.english(), s.emotions())
        self._open_session(rec)
        cancel, seconds, t0 = threading.Event(), 0.0, time.time()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16; rate=24000; channels=1")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Tilsim-Sample-Rate", "24000")
            self.end_headers()
            try:
                for ev in s.streamer.stream(text, cancel=cancel, priority=SLOT_PRIORITY["key"], **style):
                    if ev["type"] == "audio" and len(ev["pcm"]):
                        chunk = pcm16(ev["pcm"])
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                        seconds += len(ev["pcm"]) / 24000
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:  # the client hung up: stop making audio nobody hears
                cancel.set()
                self.close_connection = True
        finally:
            self._close_session()
        s.keys.record(rec, len(text), seconds)
        log(f"stream {rec['name']} {len(text)} chars -> {seconds:.1f} s audio in {time.time() - t0:.1f} s")

    def _warm(self) -> None:
        """POST /v1/warm: wakes the model (on the cloud the request itself starts the container) and reports."""
        s = self.service
        self._stream_caller()
        t0 = time.time()
        while s.voice is None and time.time() - t0 < 120:
            time.sleep(0.5)
        self._ready()
        self._json(200, {"ok": True, "ready": True, "stream": s.streamer is not None,
                         "streams": s.streamer.streams if s.streamer else 0, "waited_ms": int(1000 * (time.time() - t0))})

    def _token(self, body: dict) -> None:
        """POST /v1/stream/token {"ttl": seconds}: a browser token for /v1/stream?token=..."""
        kind, rec = self._caller()
        if kind != "key":
            raise ApiError(401, "no_key", "Token faqat API kalit bilan olinadi.")
        try:
            ttl = max(30, min(int(body.get("ttl", 900)), TOKEN_MAX_TTL))
        except (TypeError, ValueError):
            raise ApiError(400, "bad_ttl", "ttl soniyada, butun son.") from None
        token, exp = make_token(rec, ttl)
        self._json(200, {"token": token, "expires": exp, "ttl": ttl})

    def _key_limits(self, rec: dict, chars: int) -> None:
        wait = self.service.keys.rate_wait(rec)
        if wait:
            raise ApiError(429, "rate_limited", f"Daqiqasiga {rec['per_minute']} so‘rovdan oshdi. {wait} s kuting.", wait)
        left = self.service.keys.quota_left(rec)
        if left is not None and chars > left:
            raise ApiError(402, "quota_exceeded", f"Oylik limit tugadi: {left} belgi qoldi.")

    # ---------------------------------------------------------------- routes
    def do_OPTIONS(self):  # noqa: N802 (CORS preflight for the website)
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        s = self.service
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path in ("/", "/v1/health"):
                return self._json(200, {"ok": True, "ready": s.voice is not None, "busy": s.busy,
                                        "queue": s.jobs_q.qsize(), "max_chars": DEMO_MAX_CHARS, "voice": "Asoschi",
                                        "stream": s.streamer is not None, "sessions": s.sessions,
                                        "docs": "https://github.com/JAVOHIR120/tilsim/blob/main/docs/API.md"})
            if path == "/v1/stream":
                return self._websocket()
            if path == "/v1/voices":
                return self._json(200, {"voices": [{"id": vid, "sample_rate": 24000,
                                                    **{k: v for k, v in info.items() if k != "profile"},
                                                    "languages": ["uz", "en", "auto"] if s.english() else ["uz"],
                                                    "emotions": [{"id": e, "en": en} for e, en in EMOTIONS.items()]
                                                    if s.emotions() else [{"id": "neytral", "en": "neutral"}],
                                                    "default_emotion": "neytral", "intensity": {"min": 0, "max": 1,
                                                                                                "default": 0.6}}
                                                   for vid, info in VOICES.items()], "default": DEFAULT_VOICE})
            kind, rec = self._caller()
            if kind != "key":
                raise ApiError(401, "no_key", "Bu manzil faqat API kalit bilan ishlaydi.")
            if path == "/v1/usage":
                return self._json(200, s.keys.usage(rec))
            parts = path.split("/")
            if len(parts) in (4, 5) and parts[:3] == ["", "v1", "jobs"]:
                job = s.jobs.get(parts[3])
                if not job or job["key"]["id"] != rec["id"]:
                    raise ApiError(404, "no_job", "Bunday vazifa yo‘q (yoki 1 soatdan eski).")
                if len(parts) == 4:
                    return self._json(200, self._job_view(job))
                if parts[4] == "audio":
                    if job["status"] != "done":
                        raise ApiError(409, "not_ready", f"Vazifa hali tayyor emas: {job['status']}.")
                    r = job["result"]
                    return self._send(200, r["audio"], FORMATS[job["req"]["format"]],
                                      {"X-Tilsim-Seconds": f"{r['seconds']:.2f}",
                                       "Content-Disposition": f'attachment; filename="tilsim-{job["id"]}.{job["req"]["format"]}"'})
            raise ApiError(404, "not_found", "Bunday manzil yo‘q. Qo‘llanma: docs/API.md")
        except ApiError as e:
            self._error(e)

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        try:
            # read the body first: on a kept-alive connection an unread body would corrupt the next request
            body = self._body()
        except ApiError as e:
            self.close_connection = True
            return self._error(e)
        try:
            if path == "/v1/tts":
                return self._tts(body)
            if path == "/v1/tts/stream":
                return self._tts_stream(body)
            if path == "/v1/warm":
                return self._warm()
            if path == "/v1/stream/token":
                return self._token(body)
            if path == "/v1/jobs":
                return self._new_job(body)
            if path == "/v1/normalize":
                kind, _ = self._caller()
                req = self.service.prepare(body, 5000 if kind == "key" else DEMO_MAX_CHARS)
                return self._json(200, {"text": req["text"], "spoken": req["spoken"], "sentences": req["sentences"]})
            raise ApiError(404, "not_found", "Bunday manzil yo‘q. Qo‘llanma: docs/API.md")
        except ApiError as e:
            self._error(e)

    def _tts(self, body: dict) -> None:
        s = self.service
        kind, rec = self._caller()
        self._ready()
        req = s.prepare(body, SYNC_MAX_CHARS if kind == "key" else DEMO_MAX_CHARS)
        if kind == "demo":
            req["format"] = "mp3"
        who = rec["name"] if rec else self._visitor()
        cache_key, result = s.cached(req)
        hit = result is not None
        t0 = time.time()
        if not hit:
            if kind == "key":
                self._key_limits(rec, len(req["text"]))
            else:
                wait = s.demo_wait(who)
                if wait < 0:
                    raise ApiError(429, "daily_limit", "Bugungi bepul sinovlar tugadi. Ertaga urinib ko‘ring "
                                                       "yoki erta kirish ro‘yxatiga yoziling.", 3600)
                if wait:
                    raise ApiError(429, "rate_limited", f"Juda ko‘p so‘rov. {wait} soniyadan keyin urinib ko‘ring.", wait)
            fut: Future = Future()
            s.submit(kind, req, fut)
            try:
                result = fut.result(timeout=170)
            except Exception as e:  # noqa: BLE001
                log(f"{who} failed: {e!r}")
                raise ApiError(500, "failed", "Ovoz yaratishda xato bo‘ldi. Qayta urinib ko‘ring.") from None
            s.remember(cache_key, result)
        try:
            audio = encode(result["pcm"], result["sr"], req["format"])
        except Exception as e:  # noqa: BLE001
            log(f"{who} encode failed: {e!r}")
            raise ApiError(500, "failed", "Audio faylga o‘girishda xato bo‘ldi.") from None
        log(f"{who} {len(req['text'])} chars {req['format']} -> {result['seconds']:.1f} s audio in "
            f"{time.time() - t0:.1f} s{' (cache)' if hit else ''}")
        self._send(200, audio, FORMATS[req["format"]],
                   {"X-Tilsim-Seconds": f"{result['seconds']:.2f}", "X-Tilsim-Chars": str(len(req["text"])),
                    "X-Tilsim-Spoken": urllib.parse.quote(result["spoken"])})
        if rec and not hit:  # after the answer is sent: on the cloud this write is a volume commit
            s.keys.record(rec, len(req["text"]), result["seconds"])

    def _new_job(self, body: dict) -> None:
        s = self.service
        kind, rec = self._caller()
        if kind != "key":
            raise ApiError(401, "no_key", "Vazifalar faqat API kalit bilan ishlaydi.")
        self._ready()
        req = s.prepare(body, rec.get("max_chars", 5000))
        s.sweep_jobs()
        active = sum(1 for j in s.jobs.values() if j["key"]["id"] == rec["id"] and j["status"] in ("queued", "working"))
        if active >= ACTIVE_JOBS_PER_KEY:
            raise ApiError(429, "too_many_jobs", f"Bir vaqtda {ACTIVE_JOBS_PER_KEY} tadan ko‘p vazifa bo‘lmaydi.", 30)
        self._key_limits(rec, len(req["text"]))
        job = {"id": "job_" + secrets.token_hex(6), "key": rec, "req": req, "status": "queued", "done": 0,
               "total": len(req["sentences"]), "created": time.time()}
        s.submit("job", job, None)
        s.jobs[job["id"]] = job
        log(f"job {job['id']} queued ({rec['name']}): {len(req['text'])} chars, {job['total']} sentences")
        self._json(202, self._job_view(job))

    @staticmethod
    def _job_view(job: dict) -> dict:
        view = {"id": job["id"], "status": job["status"], "format": job["req"]["format"],
                "progress": {"done": job["done"], "total": job["total"]},
                "status_url": f"/v1/jobs/{job['id']}", "audio_url": f"/v1/jobs/{job['id']}/audio"}
        if job["status"] == "done":
            view["seconds"] = job["seconds"]
        if job["status"] == "failed":
            view["error"] = job.get("error")
        return view


def build(host: str, port: int, preload: bool = False, before_read=None, after_write=None) -> ThreadingHTTPServer:
    """The HTTP server, ready to serve_forever(). The cloud passes volume reload/commit hooks for the key file."""
    keys = KeyStore(KEYS_FILE, before_read=before_read, after_write=after_write)
    Handler.service = Service(keys, preload=preload)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{host}:{port}, {sum(k['active'] for k in keys.data['keys'])} active API keys"
        f"{'' if preload else ' (model loading in the background)'}")
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: the tunnel reaches it locally")
    args = ap.parse_args()
    set_priority_above_normal()
    build(args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
