"""Live speech sessions: text arrives in pieces (from an LLM), audio leaves in pieces, the listener can interrupt.

Transport-free, so it is tested without a GPU or a socket (tests/test_stream_api.py). engine/serve.py wires it to
a WebSocket (/v1/stream) and to a chunked HTTP response (/v1/tts/stream).

Client -> session:  feed(text, flush=False) · flush() · cancel() · close()
Session -> client:  send_event({"type": "start" | "mark" | "end" | "error", ...}) and send_audio(PCM16 bytes)

Turn: everything fed until a flush. Sentences start speaking as soon as they are complete, before the flush.
After the flush, and once the last sentence is spoken, "end" is sent.
cancel() stops the sentence being spoken within one token (~15 ms), drops the queued ones and sends
{"type": "end", "reason": "cancelled"}. The session stays open for the next turn.
"""
from __future__ import annotations

import queue
import re
import threading
import time

import numpy as np

SR = 24000
PAUSE = 0.25
FLUSH_MARK = object()
# A sentence ends at . ! ? … (one or more), optionally followed by closing quotes/brackets, then whitespace and a
# word that does not start in lower case (so «U «Rahmat!» dedi.» stays one sentence). Until that next word has
# arrived the end is not certain, so the sentence waits for it (one LLM token) or for a flush.
_END = re.compile(r"""[.!?…]+["'»”’)\]]*(?=\s+[^\s\w]*[^\W\d_a-zà-ÿ'‘’ʻ])|[.!?…]+["'»”’)\]]*(?=\s+\d)""")
_SOFT = re.compile(r"[,;:—–]\s")
SOFT_SPLIT = 120   # chars: with no sentence end yet, speak up to the last comma so speech starts sooner
HARD_SPLIT = 220   # chars: cut at the last space no matter what


class SlotPool:
    """Parallel-stream slots handed out by priority (lower first), first come first served within a priority."""

    def __init__(self, slots: list):
        import heapq
        import itertools
        self._heapq, self._seq = heapq, itertools.count()
        self.free = list(slots)
        self.cv = threading.Condition()
        self.waiting: list = []

    def acquire(self, priority: int = 1):
        with self.cv:
            me = (priority, next(self._seq))
            self._heapq.heappush(self.waiting, me)
            while not (self.free and self.waiting[0] == me):
                self.cv.wait()
            self._heapq.heappop(self.waiting)
            slot = self.free.pop()
            self.cv.notify_all()
            return slot

    def release(self, slot) -> None:
        with self.cv:
            self.free.append(slot)
            self.cv.notify_all()


class SentenceBuffer:
    """Collects streamed text and hands out whole sentences as soon as they are complete."""

    def __init__(self):
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            m = _END.search(self.buf)
            if m:
                out.append(self.buf[:m.end()].strip())
                self.buf = self.buf[m.end():].lstrip()
                continue
            if len(self.buf) >= SOFT_SPLIT:
                soft = list(_SOFT.finditer(self.buf[:HARD_SPLIT]))
                cut = soft[-1].end() if soft else (self.buf.rfind(" ", 0, HARD_SPLIT) + 1 if len(self.buf) >= HARD_SPLIT else 0)
                if cut > 0:
                    out.append(self.buf[:cut].strip())
                    self.buf = self.buf[cut:].lstrip()
                    continue
            break
        return [s for s in out if s]

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


def pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class LiveSession:
    """One listener. A worker thread speaks queued sentences; feed/flush/cancel may come from another thread."""

    def __init__(self, streamer, send_event, send_audio, max_chars: int | None = None, on_usage=None,
                 priority: int = 0, seed: int = 0):
        self.streamer, self.send_event, self.send_audio = streamer, send_event, send_audio
        self.max_chars, self.on_usage, self.priority, self.seed = max_chars, on_usage, priority, seed
        self.buffer = SentenceBuffer()
        self.q: queue.Queue = queue.Queue()
        self.cancel_ev = threading.Event()
        self.lock = threading.Lock()
        self.turn = 0                 # bumps on cancel: queued items of an old turn are dropped
        self.started = False          # "start" sent for the current turn
        self.spoken_in_turn = 0
        self.chars = 0                  # queued in this session (quota check)
        self.seconds = 0.0              # audio sent in this session
        self.billed_chars_pending = 0   # spoken since the last bill
        self.billed_seconds = 0.0
        self.closed = False
        self.pacer = None               # engine/stream.Pacer for the current turn (shared by its sentences)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------ client side
    def feed(self, text: str, flush: bool = False) -> None:
        with self.lock:
            for s in self.buffer.feed(text):
                self._queue(s)
        if flush:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            for s in self.buffer.flush():
                self._queue(s)
            self.q.put((self.turn, FLUSH_MARK))

    def cancel(self) -> None:
        with self.lock:
            self.turn += 1
            self.buffer = SentenceBuffer()
            self.cancel_ev.set()
            while True:  # drop queued sentences of the cancelled turn
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put((self.turn, "__cancelled__"))

    def close(self) -> None:
        self.closed = True
        self.cancel_ev.set()
        self.q.put((None, None))
        self.worker.join(timeout=5)
        self.bill()

    def _queue(self, sentence: str) -> None:
        if self.max_chars is not None and self.chars + len(sentence) > self.max_chars:
            self.q.put((self.turn, ("__error__", "quota_exceeded")))
            return
        self.chars += len(sentence)
        self.q.put((self.turn, sentence))

    # ------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            turn, item = self.q.get()
            if item is None:
                return
            if turn != self.turn:
                continue  # belongs to a cancelled turn
            if item == "__cancelled__":
                self.cancel_ev.clear()
                self._end("cancelled")
                continue
            if isinstance(item, tuple) and item[0] == "__error__":
                self.send_event({"type": "error", "code": item[1]})
                continue
            if item is FLUSH_MARK:
                self._end("done")
                continue
            self._speak(item, turn)

    def _new_pacer(self):
        try:
            from stream import Pacer
        except ImportError:  # tests without torch: fake streamers pace themselves
            return None
        return Pacer()

    def _end(self, reason: str) -> None:
        """Every flush and every cancel gets exactly one "end", so a client never waits forever."""
        self.send_event({"type": "end", "reason": reason})
        self.started = False
        self.spoken_in_turn = 0
        self.bill()

    def bill(self) -> None:
        """Hand the characters spoken and audio seconds sent since the last bill to on_usage (once per turn:
        on the cloud every usage write is a volume commit, too slow to do per sentence)."""
        chars, secs = self.billed_chars_pending, self.seconds - self.billed_seconds
        self.billed_chars_pending, self.billed_seconds = 0, self.seconds
        if self.on_usage and (chars or secs > 0):
            self.on_usage(chars, secs)

    def _speak(self, sentence: str, turn: int) -> None:
        if not self.started:
            self.send_event({"type": "start"})
            self.started = True
            self.pacer = self._new_pacer()
        elif self.spoken_in_turn:
            self.send_audio(pcm16(np.zeros(int(PAUSE * SR), np.float32)))
            self.seconds += PAUSE
            if self.pacer is not None:
                self.pacer.emitted(PAUSE)
        t0 = time.perf_counter()
        try:
            kw = {"pacer": self.pacer} if self.pacer is not None else {}
            for ev in self.streamer.stream(sentence, cancel=self.cancel_ev, seed=self.seed + self.spoken_in_turn,
                                           priority=self.priority, **kw):
                if self.cancel_ev.is_set() or turn != self.turn or self.closed:
                    return
                if ev["type"] == "mark":
                    self.send_event({"type": "mark", "text": ev["text"]})
                elif ev["type"] == "audio" and len(ev["pcm"]):
                    self.send_audio(pcm16(ev["pcm"]))
                    self.seconds += len(ev["pcm"]) / SR
        except Exception:  # noqa: BLE001
            self.send_event({"type": "error", "code": "failed"})
            return
        self.spoken_in_turn += 1
        self.billed_chars_pending += len(sentence)
        self.last_sentence_seconds = time.perf_counter() - t0
