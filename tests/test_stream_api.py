"""Live streaming without a GPU: sentence buffer, LiveSession, slot priority, and the WebSocket / chunked HTTP API
on a real socket with a fake voice.

    python -m unittest tests.test_stream_api -v
"""
import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT)]

import apikeys  # noqa: E402
import live  # noqa: E402
import serve  # noqa: E402

SR = 24000


class FakeStreamer:
    """Speaks every sentence as `chunks` pieces of 0.1 s tone, `delay` seconds apart (like a GPU making them).
    Languages and emotions are only recorded (self.styles), the way engine/stream.Streamer.style() would pick them."""
    sr, streams = SR, 1
    EMOTIONS = ("neytral", "iliq", "quvnoq", "hayajonli", "xotirjam", "jiddiy", "hamdard", "hayron")

    def __init__(self, chunks=3, delay=0.01):
        self.chunks, self.delay = chunks, delay
        self.calls = []
        self.styles = []          # (sentence, lang, emotion, intensity) per sentence spoken
        self.english = object()   # English mode "loaded"; a test sets None to switch it off
        self.bank = {"iliq": {}}    # an emotion bank exists; a test empties it to switch emotions off

    def plan(self, text, lang="uz"):
        from uznorm import split_sentences
        from uznorm.lang import segments
        if lang == "auto" and self.english is not None:
            return segments(text)
        return [("en" if lang == "en" else "uz", s) for s in split_sentences(text.strip(), max_chars=10_000)]

    def style(self, lang="uz", emotion=None, intensity=None):
        emotion = emotion or "neytral"
        assert lang in ("uz", "en") and emotion in self.EMOTIONS
        return {"lang": lang, "emotion": emotion, "intensity": intensity}

    def stream(self, text, cancel=None, seed=0, priority=1, lang="uz", emotion=None, intensity=None, **kw):
        sents = self.plan(text, lang)
        for i, (lg, sent) in enumerate(sents):
            st = self.style(lg, emotion, intensity)
            self.calls.append((sent, priority))
            yield {"type": "mark", "text": sent, "index": i, "lang": lg, "emotion": st["emotion"]}
            for _ in self.sentence(sent, seed, cancel, priority, style=st):
                yield {"type": "audio", "pcm": _}
            if i < len(sents) - 1:  # the real Streamer pauses 0.25 s between sentences
                yield {"type": "audio", "pcm": np.zeros(int(0.25 * SR), np.float32)}

    def sentence(self, spoken, seed=0, cancel=None, priority=1, pacer=None, whole=False, style=None):
        st = style or self.style()
        self.styles.append((spoken, st["lang"], st["emotion"], st["intensity"]))
        for _ in range(self.chunks):
            if cancel is not None and cancel.is_set():
                return
            time.sleep(self.delay)
            yield 0.1 * np.sin(np.arange(int(0.1 * SR)) * 2 * np.pi * 220 / SR).astype(np.float32)


class SentenceBufferTest(unittest.TestCase):
    def test_sentences_as_they_complete(self):
        b = live.SentenceBuffer()
        self.assertEqual(b.feed("Salom! Endi Part"), ["Salom!"])
        self.assertEqual(b.feed(" 2 ga o‘tamiz. Describe"), ["Endi Part 2 ga o‘tamiz."])
        self.assertEqual(b.feed(" a place you like to visit."), [])  # no space after the dot yet
        self.assertEqual(b.flush(), ["Describe a place you like to visit."])

    def test_long_text_without_end_splits_at_a_comma(self):
        b = live.SentenceBuffer()
        text = "Bu juda uzun gap, " * 8
        out = b.feed(text)
        self.assertTrue(out and out[0].endswith(","))
        self.assertLessEqual(len(out[0]), live.HARD_SPLIT)

    def test_quotes_after_the_end(self):
        b = live.SentenceBuffer()
        self.assertEqual(b.feed("U «Rahmat!» dedi. Keyin"), ["U «Rahmat!» dedi."])


class SlotPoolTest(unittest.TestCase):
    def test_lower_priority_number_goes_first(self):
        pool = live.SlotPool(["slot"])
        held = pool.acquire(1)
        order = []

        def take(p, name):
            s = pool.acquire(p)
            order.append(name)
            pool.release(s)
        t_job = threading.Thread(target=take, args=(3, "job"))
        t_job.start()
        time.sleep(0.05)
        t_live = threading.Thread(target=take, args=(0, "live"))
        t_live.start()
        time.sleep(0.05)
        pool.release(held)
        t_job.join(2)
        t_live.join(2)
        self.assertEqual(order, ["live", "job"])


class LiveSessionTest(unittest.TestCase):
    def make(self, **kw):
        events, audio, usage = [], [], []
        s = live.LiveSession(kw.pop("streamer", FakeStreamer()), events.append, audio.append,
                             on_usage=lambda c, sec: usage.append((c, sec)), **kw)
        return s, events, audio, usage

    def wait_end(self, events, n=1, timeout=5):
        t0 = time.time()
        while sum(e["type"] == "end" for e in events) < n and time.time() - t0 < timeout:
            time.sleep(0.01)

    def test_slow_billing_does_not_delay_the_next_turn(self):
        """On the cloud a usage write takes ~2 s (volume commit); the next turn must not wait for it."""
        events, audio, usage = [], [], []

        def slow_usage(c, sec):
            time.sleep(1.0)
            usage.append((c, sec))
        s = live.LiveSession(FakeStreamer(chunks=2), events.append, audio.append, on_usage=slow_usage)
        s.feed("Birinchi navbat.", flush=True)
        self.wait_end(events)
        t0 = time.time()
        s.feed("Ikkinchi navbat.", flush=True)
        while sum(e["type"] == "start" for e in events) < 2 and time.time() - t0 < 3:
            time.sleep(0.005)
        self.assertLess(time.time() - t0, 0.5)
        self.wait_end(events, 2)
        s.close()
        self.assertEqual([u[0] for u in usage], [len("Birinchi navbat."), len("Ikkinchi navbat.")])

    def test_turn_events_and_usage(self):
        s, events, audio, usage = self.make()
        s.feed("Salom! Bu sinov.", flush=True)
        self.wait_end(events)
        s.close()
        types = [e["type"] for e in events]
        self.assertEqual(types, ["start", "mark", "mark", "end"])
        self.assertEqual([e["text"] for e in events if e["type"] == "mark"], ["Salom!", "Bu sinov."])
        self.assertEqual(events[-1]["reason"], "done")
        self.assertEqual(len(audio), 3 + 1 + 3)  # 3 chunks, the pause between sentences, 3 chunks
        self.assertTrue(all(isinstance(a, bytes) and len(a) % 2 == 0 for a in audio))
        self.assertEqual(usage[0][0], len("Salom!") + len("Bu sinov."))
        self.assertAlmostEqual(usage[0][1], 0.6 + live.PAUSE, places=2)

    def test_cancel_stops_fast_and_next_turn_works(self):
        s, events, audio, _ = self.make(streamer=FakeStreamer(chunks=50, delay=0.02))
        s.feed("Juda uzun javob boshlandi. Ikkinchi gap ham bor.", flush=True)
        t_wait = time.time()
        while not audio and time.time() - t_wait < 5:
            time.sleep(0.005)
        self.assertTrue(audio, f"no audio, events: {events}")
        t0 = time.time()
        s.cancel()
        self.wait_end(events)
        self.assertLess(time.time() - t0, 0.5)
        self.assertEqual(events[-1], {"type": "end", "reason": "cancelled"})
        self.assertLess(len(audio), 10)
        n = len(events)
        s.feed("Yangi savol.", flush=True)
        self.wait_end(events, 2)
        s.close()
        self.assertEqual([e["type"] for e in events[n:]], ["start", "mark", "end"])

    def test_live_sessions_ask_for_the_top_priority(self):
        fake = FakeStreamer()
        s, events, _, _ = self.make(streamer=fake, priority=0)
        s.feed("Salom.", flush=True)
        self.wait_end(events)
        s.close()
        self.assertEqual(fake.calls[0][1], 0)

    def test_quota(self):
        s, events, _, _ = self.make(max_chars=10)
        s.feed("Bu gap o‘ndan uzun.", flush=True)
        self.wait_end(events)
        s.close()
        self.assertIn({"type": "error", "code": "quota_exceeded"}, events)

    def test_empty_flush_still_ends(self):
        s, events, _, _ = self.make()
        s.flush()
        self.wait_end(events)
        s.close()
        self.assertEqual(events, [{"type": "end", "reason": "done"}])


# ---------------------------------------------------------------- a tiny WebSocket client for the tests
class WSClient:
    def __init__(self, port, path, headers=None):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        lines = [f"GET {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Upgrade: websocket", "Connection: Upgrade",
                 f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"] + [f"{k}: {v}" for k, v in (headers or {}).items()]
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        self.f = self.sock.makefile("rb")
        status = self.f.readline().decode()
        self.status = int(status.split()[1])
        self.headers = {}
        while True:
            ln = self.f.readline().decode().strip()
            if not ln:
                break
            k, v = ln.split(":", 1)
            self.headers[k.strip().lower()] = v.strip()
        if self.status != 101:
            n = int(self.headers.get("content-length", "0"))
            self.body = json.loads(self.f.read(n)) if n else {}
        else:
            import wsproto
            assert self.headers["sec-websocket-accept"] == wsproto.accept_key(key)

    def send(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        head = struct.pack("!BB", 0x81, 0x80 | n) if n < 126 else struct.pack("!BBH", 0x81, 0x80 | 126, n)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        b1, b2 = self.f.read(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self.f.read(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self.f.read(8))[0]
        data = self.f.read(n)
        op = b1 & 0x0F
        return ("text", json.loads(data)) if op == 1 else ("binary", data) if op == 2 else ("op", op)

    def until(self, kind="end", limit=500):
        got = []
        for _ in range(limit):
            k, v = self.recv()
            got.append((k, v))
            if k == "text" and v.get("type") == kind:
                return got
        raise AssertionError(f"no {kind} in {got[-5:]}")

    def close(self):
        self.sock.close()


class StreamService(serve.Service):
    def _load(self):
        self.streamer = FakeStreamer()
        self.voice = serve.StreamVoice(self.streamer)


class StreamServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        serve.LOG = Path(cls.dir.name) / "serve.log"
        cls.store = apikeys.KeyStore(Path(cls.dir.name) / "keys.json")
        cls.rec, cls.key = cls.store.create("ielts", per_minute=1000)
        handler = type("H", (serve.Handler,), {"service": StreamService(cls.store)})
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        for _ in range(100):
            if handler.service.voice is not None:
                break
            time.sleep(0.02)
        cls.handler = handler

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.dir.cleanup()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def test_ws_needs_a_key(self):
        c = WSClient(self.port, "/v1/stream")
        self.assertEqual((c.status, c.body["code"]), (401, "no_key"))
        c.close()
        c = WSClient(self.port, "/v1/stream?key=tlsm_wrong")
        self.assertEqual((c.status, c.body["code"]), (401, "bad_key"))
        c.close()

    def test_ws_turn_with_header_key(self):
        c = WSClient(self.port, "/v1/stream", {"X-API-Key": self.key})
        self.assertEqual(c.status, 101)
        self.assertEqual(c.recv()[1]["type"], "ready")
        c.send({"type": "text", "text": "Salom! Endi Part 2 ga o‘tamiz.", "flush": False})
        c.send({"type": "flush"})
        got = c.until("end")
        kinds = [k if k == "binary" else v["type"] for k, v in got]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds.count("mark"), 2)
        self.assertGreaterEqual(kinds.count("binary"), 6)
        self.assertEqual(got[-1][1]["reason"], "done")
        pcm = b"".join(v for k, v in got if k == "binary")
        self.assertEqual(len(pcm) % 2, 0)  # PCM16
        c.close()

    def test_ws_cancel(self):
        self.handler.service.streamer.chunks, self.handler.service.streamer.delay = 60, 0.02
        try:
            c = WSClient(self.port, f"/v1/stream?key={self.key}")
            c.recv()
            c.send({"type": "text", "text": "Juda uzun javob. Yana bir gap.", "flush": True})
            while c.recv()[0] != "binary":
                pass
            c.send({"type": "cancel"})
            got = c.until("end")
            self.assertEqual(got[-1][1]["reason"], "cancelled")
            self.assertLess(sum(k == "binary" for k, _ in got), 30)
            c.close()
        finally:
            self.handler.service.streamer.chunks, self.handler.service.streamer.delay = 3, 0.01

    def test_ws_browser_token(self):
        status, _, body = self.post("/v1/stream/token", {"ttl": 5}, {"X-API-Key": self.key})
        tok = json.loads(body)
        self.assertEqual((status, tok["ttl"]), (200, 30))  # clamped to at least 30 s
        c = WSClient(self.port, f"/v1/stream?token={tok['token']}")
        self.assertEqual(c.status, 101)
        c.close()
        forged = tok["token"][:-3] + ("AAA" if not tok["token"].endswith("AAA") else "BBB")
        c = WSClient(self.port, f"/v1/stream?token={forged}")
        self.assertEqual((c.status, c.body["code"]), (401, "bad_token"))
        c.close()
        expired, _ = serve.make_token(self.rec, -5)
        c = WSClient(self.port, f"/v1/stream?token={expired}")
        self.assertEqual(c.status, 401)
        c.close()

    def test_ws_bad_messages(self):
        c = WSClient(self.port, "/v1/stream", {"X-API-Key": self.key})
        c.recv()
        c.send({"type": "dance"})
        self.assertEqual(c.recv()[1]["code"], "bad_type")
        c.close()

    def test_ws_usage_is_recorded(self):
        _, key = self.store.create("usage-only", per_minute=1000)  # its own key: other tests bill asynchronously
        before = self.store.usage(self.store.find(key))["chars"]
        c = WSClient(self.port, "/v1/stream", {"X-API-Key": key})
        c.recv()
        c.send({"type": "text", "text": "Hisob.", "flush": True})
        c.until("end")
        c.close()
        time.sleep(0.1)
        for _ in range(50):
            if self.store.usage(self.store.find(key))["chars"] > before:
                break
            time.sleep(0.05)
        self.assertEqual(self.store.usage(self.store.find(key))["chars"], before + len("Hisob."))

    def test_http_stream_is_chunked_pcm(self):
        req = urllib.request.Request(self.base + "/v1/tts/stream", data=json.dumps({"text": "Birinchi gap shu yerda tugaydi va biroz uzunroq bo‘ladi. Ikkinchi gap esa mana shu yerda tugaydi."}).encode(),
                                     method="POST", headers={"Content-Type": "application/json", "X-API-Key": self.key})
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.headers["Transfer-Encoding"], "chunked")
            self.assertTrue(r.headers["Content-Type"].startswith("audio/L16"))
            pcm = r.read()
        self.assertEqual(len(pcm), 2 * int(SR * (0.6 + 0.25)))  # 2 sentences x 3 chunks + one pause

    def test_http_stream_needs_a_key(self):
        status, _, body = self.post("/v1/tts/stream", {"text": "Salom."}, {"Origin": "https://javohir120.github.io"})
        self.assertEqual((status, json.loads(body)["code"]), (401, "no_key"))

    def test_warm(self):
        status, _, body = self.post("/v1/warm", {}, {"X-API-Key": self.key})
        out = json.loads(body)
        self.assertEqual((status, out["ready"], out["stream"]), (200, True, True))

    def test_sessions_are_capped(self):
        svc = self.handler.service
        for _ in range(300):   # sessions of earlier tests close on the server a moment after the client hangs up
            if svc.sessions <= 0:
                break
            time.sleep(0.01)
        svc.sessions = serve.MAX_SESSIONS_PER_STREAM * svc.streamer.streams
        try:
            c = WSClient(self.port, "/v1/stream", {"X-API-Key": self.key})
            self.assertEqual((c.status, c.body["code"]), (503, "busy"))
            c.close()
        finally:
            svc.sessions = 0

    def test_tts_runs_on_the_stream_engine(self):
        svc = self.handler.service
        svc.streamer.calls.clear()
        status, h, _ = self.post("/v1/tts", {"text": "Oddiy so‘rov ham shu ovoz bilan ishlaydi.", "format": "wav"},
                                 {"X-API-Key": self.key})
        self.assertEqual((status, h["Content-Type"]), (200, "audio/wav"))

    # ------------------------------------------------------------ language and emotion (2026-11-15)
    def test_parse_style(self):
        ps = serve.parse_style
        self.assertEqual(ps({}, True), {})
        self.assertEqual(ps({"lang": "EN", "emotion": "Iliq", "intensity": "0.8"}, True),
                         {"lang": "en", "emotion": "iliq", "intensity": 0.8})
        self.assertEqual(ps({"intensity": 0}, True), {"intensity": 0.0})
        for body, code, english in [({"lang": "ru"}, "bad_lang", True), ({"lang": "en"}, "lang_unavailable", False),
                                    ({"emotion": "g‘azab"}, "bad_emotion", True), ({"intensity": 1.5}, "bad_intensity", True),
                                    ({"intensity": -0.1}, "bad_intensity", True), ({"intensity": "nan"}, "bad_intensity", True),
                                    ({"intensity": "kuchli"}, "bad_intensity", True)]:
            with self.assertRaises(serve.ApiError) as cm:
                ps(body, english)
            self.assertEqual(cm.exception.code, code, body)
        self.assertEqual(ps({"lang": "auto"}, False), {"lang": "auto"})  # auto without English: all Uzbek
        with self.assertRaises(serve.ApiError) as cm:                     # no emotion bank yet: refused, not ignored
            ps({"emotion": "iliq"}, True, emotions=False)
        self.assertEqual(cm.exception.code, "emotion_unavailable")
        self.assertEqual(ps({"emotion": "neytral"}, True, emotions=False), {"emotion": "neytral"})

    def test_plain_style_is_the_old_path(self):
        self.assertTrue(serve.is_plain({}))
        self.assertTrue(serve.is_plain({"lang": "uz", "emotion": "neytral", "intensity": 1}))
        self.assertFalse(serve.is_plain({"emotion": "iliq"}))
        self.assertFalse(serve.is_plain({"lang": "auto"}))

    def test_ws_style_is_sticky_and_marks_name_it(self):
        st = self.handler.service.streamer
        st.styles.clear()
        c = WSClient(self.port, f"/v1/stream?key={self.key}")
        c.recv()
        c.send({"type": "text", "text": "Salom! ", "emotion": "iliq", "intensity": 0.8})
        c.send({"type": "text", "text": "Endi Part 2 ga o‘tamiz.", "flush": True})   # no fields: style stays
        got = c.until("end")
        marks = [v for k, v in got if k == "text" and v["type"] == "mark"]
        self.assertEqual([(m["lang"], m["emotion"]) for m in marks], [("uz", "iliq"), ("uz", "iliq")])
        self.assertEqual([s[2:] for s in st.styles], [("iliq", 0.8), ("iliq", 0.8)])
        c.send({"type": "text", "text": "Yaxshi.", "emotion": "neytral", "flush": True})   # changed back
        got = c.until("end")
        self.assertEqual([v["emotion"] for k, v in got if k == "text" and v["type"] == "mark"], ["neytral"])
        c.close()

    def test_ws_bad_style_skips_the_message_only(self):
        c = WSClient(self.port, f"/v1/stream?key={self.key}")
        c.recv()
        c.send({"type": "text", "text": "Bu aytilmaydi.", "emotion": "g‘azab", "flush": True})
        k, v = c.recv()
        self.assertEqual((v["type"], v["code"]), ("error", "bad_emotion"))
        c.send({"type": "text", "text": "Bu aytiladi.", "flush": True})
        got = c.until("end")
        self.assertEqual([v["text"] for k, v in got if k == "text" and v["type"] == "mark"], ["Bu aytiladi."])
        c.close()

    def test_ws_auto_language_per_sentence(self):
        c = WSClient(self.port, f"/v1/stream?key={self.key}")
        c.recv()
        c.send({"type": "text", "text": "Javobingiz uchun rahmat. That's the end of the speaking test.",
                "lang": "auto", "flush": True})
        got = c.until("end")
        marks = [(v["lang"], v["text"]) for k, v in got if k == "text" and v["type"] == "mark"]
        self.assertEqual(marks, [("uz", "Javobingiz uchun rahmat."), ("en", "That's the end of the speaking test.")])
        c.close()

    def test_tts_with_emotion_and_english(self):
        st = self.handler.service.streamer
        st.styles.clear()
        status, h, _ = self.post("/v1/tts", {"text": "Bugun ajoyib kun! Keling, boshlaymiz.", "emotion": "quvnoq",
                                             "intensity": 0.5, "format": "wav", "seed": 11}, {"X-API-Key": self.key})
        self.assertEqual((status, h["Content-Type"]), (200, "audio/wav"))
        self.assertTrue(st.styles and all(s[1:] == ("uz", "quvnoq", 0.5) for s in st.styles), st.styles)
        st.styles.clear()
        status, h, _ = self.post("/v1/tts", {"text": "The test starts at 9:30.", "lang": "en", "format": "wav",
                                             "seed": 12}, {"X-API-Key": self.key})
        self.assertEqual(status, 200)
        # (the fake's audio is too short for its text, so StreamVoice retakes it: same style every time)
        self.assertEqual(set(st.styles), {("The test starts at nine thirty.", "en", "neytral", None)})
        status, _, body = self.post("/v1/tts", {"text": "Salom.", "emotion": "g‘azab"}, {"X-API-Key": self.key})
        self.assertEqual((status, json.loads(body)["code"]), (400, "bad_emotion"))

    def test_voices_lists_languages_and_emotions(self):
        with urllib.request.urlopen(self.base + "/v1/voices", timeout=5) as r:
            v = json.loads(r.read())["voices"][0]
        self.assertEqual(v["languages"], ["uz", "en", "auto"])
        self.assertEqual([e["id"] for e in v["emotions"]],
                         ["neytral", "iliq", "quvnoq", "hayajonli", "xotirjam", "jiddiy", "hamdard", "hayron"])
        self.assertEqual((v["default_emotion"], v["intensity"]["default"]), ("neytral", 0.6))

    def test_no_emotion_bank_is_refused_cleanly(self):
        st = self.handler.service.streamer
        saved, st.bank = st.bank, {}
        try:
            status, _, body = self.post("/v1/tts", {"text": "Salom.", "emotion": "quvnoq"}, {"X-API-Key": self.key})
            self.assertEqual((status, json.loads(body)["code"]), (400, "emotion_unavailable"))
            with urllib.request.urlopen(self.base + "/v1/voices", timeout=5) as r:
                v = json.loads(r.read())["voices"][0]
            self.assertEqual([e["id"] for e in v["emotions"]], ["neytral"])
            c = WSClient(self.port, f"/v1/stream?key={self.key}")
            c.recv()
            c.send({"type": "text", "text": "Salom.", "emotion": "iliq", "flush": True})
            self.assertEqual(c.recv()[1]["code"], "emotion_unavailable")
            c.close()
            for _ in range(200):   # the server closes its side a moment later; later tests count open sessions
                if self.handler.service.sessions == 0:
                    break
                time.sleep(0.01)
        finally:
            st.bank = saved

    def test_english_off_is_refused_cleanly(self):
        st = self.handler.service.streamer
        saved, st.english = st.english, None
        try:
            status, _, body = self.post("/v1/tts/stream", {"text": "Hello there.", "lang": "en"}, {"X-API-Key": self.key})
            self.assertEqual((status, json.loads(body)["code"]), (400, "lang_unavailable"))
        finally:
            st.english = saved


if __name__ == "__main__":
    unittest.main()
