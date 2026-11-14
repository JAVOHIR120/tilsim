"""Tilsim API without the GPU: the key store, and the HTTP server with a fake voice (zeros of the right length).

    python -m unittest tests.test_api -v
"""
import json
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
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))

import apikeys  # noqa: E402
import serve  # noqa: E402


class KeyStoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "keys.json"
        self.store = apikeys.KeyStore(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_create_find_revoke(self):
        rec, key = self.store.create("bot")
        self.assertTrue(key.startswith("tlsm_"))
        self.assertEqual(self.store.find(key)["id"], rec["id"])
        self.assertIsNone(self.store.find(key + "x"))
        self.assertIsNone(self.store.find("not-a-key"))
        self.assertIsNone(self.store.find(None))
        self.assertTrue(self.store.revoke(rec["id"]))
        self.assertIsNone(self.store.find(key))

    def test_plain_key_never_stored(self):
        _, key = self.store.create("bot")
        self.assertNotIn(key, self.path.read_text(encoding="utf-8"))

    def test_survives_reload(self):
        rec, key = self.store.create("bot", per_minute=5, monthly_chars=100)
        self.store.record(rec, 40, 3.5)
        again = apikeys.KeyStore(self.path)
        found = again.find(key)
        self.assertEqual(found["usage"][apikeys.month_now()]["chars"], 40)
        self.assertEqual(again.quota_left(found), 60)

    def test_rate_limit_per_minute(self):
        rec, _ = self.store.create("bot", per_minute=2)
        t = 1000.0
        self.assertEqual(self.store.rate_wait(rec, t), 0)
        self.assertEqual(self.store.rate_wait(rec, t + 1), 0)
        self.assertGreater(self.store.rate_wait(rec, t + 2), 0)
        self.assertEqual(self.store.rate_wait(rec, t + 61), 0)

    def test_no_quota_means_unlimited(self):
        rec, _ = self.store.create("owner")
        self.assertIsNone(self.store.quota_left(rec))

    def test_two_writers_do_not_undo_each_other(self):
        """The server (usage) and the key CLI (new key) write the same file; both changes must survive."""
        server = apikeys.KeyStore(self.path)
        rec, key = server.create("server-key")
        cli = apikeys.KeyStore(self.path)
        _, new_key = cli.create("made-by-cli")
        server.record(server.find(key), 10, 1.0)          # server writes after the CLI did
        again = apikeys.KeyStore(self.path)
        self.assertIsNotNone(again.find(new_key))
        self.assertEqual(again.find(key)["usage"][apikeys.month_now()]["chars"], 10)

    def test_new_and_revoked_keys_reach_a_running_server(self):
        server = apikeys.KeyStore(self.path)
        cli = apikeys.KeyStore(self.path)
        rec, key = cli.create("late")
        server.loaded_at -= 10  # an unknown key triggers a re-read (at most every 5 s)
        self.assertIsNotNone(server.find(key))
        cli.revoke(rec["id"])
        server.loaded_at -= apikeys.KeyStore.REFRESH + 1
        self.assertIsNone(server.find(key))

    def test_hooks_run_around_reads_and_writes(self):
        calls = []
        store = apikeys.KeyStore(self.path, before_read=lambda: calls.append("reload"),
                                 after_write=lambda: calls.append("commit"))
        store.create("x")
        self.assertIn("reload", calls)
        self.assertEqual(calls[-1], "commit")


class FakeVoice:
    """Stands in for the GPU model: a quiet tone, 0.05 s per letter, so lengths look real.
    The text "jimlik" gives pure digital silence (ffmpeg's loudnorm used to crash on that)."""
    sr = 24000

    def synth(self, sentences, seed=0, progress=None):
        out = []
        for i, s in enumerate(sentences):
            n = int(self.sr * 0.05 * max(1, len(s)))
            tone = 0.1 * np.sin(np.arange(n) * 2 * np.pi * 220 / self.sr).astype(np.float32)
            out.append(np.zeros(n, np.float32) if "jimlik" in s else tone)
            if progress:
                progress(i + 1)
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class FakeService(serve.Service):
    def _load(self):  # the real worker loop runs; only the model is fake
        self.voice = FakeVoice()


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        serve.LOG = Path(cls.dir.name) / "serve.log"
        store = apikeys.KeyStore(Path(cls.dir.name) / "keys.json")
        _, cls.key = store.create("test", per_minute=100)
        _, cls.slow_key = store.create("slow", per_minute=1)
        rec, cls.dead_key = store.create("dead")
        store.revoke(rec["id"])
        serve.Handler.service = FakeService(store)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        for _ in range(50):
            if serve.Handler.service.voice is not None:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.dir.cleanup()

    def call(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def hdr(self, k=None):
        return {"X-API-Key": k or self.key}

    def test_health_is_public(self):
        status, _, body = self.call("GET", "/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ready"])

    def test_voices(self):
        status, _, body = self.call("GET", "/v1/voices")
        self.assertEqual(json.loads(body)["voices"][0]["id"], "asoschi")

    def test_no_key_no_origin_refused(self):
        status, _, body = self.call("POST", "/v1/tts", {"text": "Salom."})
        self.assertEqual((status, json.loads(body)["code"]), (401, "no_key"))

    def test_bad_and_revoked_keys(self):
        for k in ("tlsm_wrong", self.dead_key):
            status, _, body = self.call("POST", "/v1/tts", {"text": "Salom."}, self.hdr(k))
            self.assertEqual((status, json.loads(body)["code"]), (401, "bad_key"))

    def test_bearer_header_works(self):
        status, h, _ = self.call("POST", "/v1/tts", {"text": "Bearer sinovi."}, {"Authorization": f"Bearer {self.key}"})
        self.assertEqual(status, 200)
        self.assertEqual(h["Content-Type"], "audio/mpeg")

    def test_formats(self):
        for fmt, ctype, magic in (("mp3", "audio/mpeg", None), ("wav", "audio/wav", b"RIFF"), ("ogg", "audio/ogg", b"OggS")):
            status, h, body = self.call("POST", "/v1/tts", {"text": f"Format {fmt}.", "format": fmt}, self.hdr())
            self.assertEqual((status, h["Content-Type"]), (200, ctype), fmt)
            if magic:
                self.assertTrue(body.startswith(magic), fmt)
            self.assertGreater(float(h["X-Tilsim-Seconds"]), 0)

    def test_other_format_comes_from_cache(self):
        before = json.loads(self.call("GET", "/v1/usage", headers=self.hdr())[2])["requests"]
        text = "Kesh sinovi: bir xil matn, boshqa format."
        a = self.call("POST", "/v1/tts", {"text": text, "format": "mp3"}, self.hdr())
        b = self.call("POST", "/v1/tts", {"text": text, "format": "wav"}, self.hdr())
        after = json.loads(self.call("GET", "/v1/usage", headers=self.hdr())[2])["requests"]
        self.assertEqual((a[0], b[0], b[1]["Content-Type"]), (200, 200, "audio/wav"))
        self.assertEqual(after, before + 1)  # the WAV needed no new GPU work
        self.assertEqual(a[1]["X-Tilsim-Seconds"], b[1]["X-Tilsim-Seconds"])

    def test_silence_does_not_crash(self):
        for fmt in ("mp3", "wav", "ogg"):
            status, _, _ = self.call("POST", "/v1/tts", {"text": "jimlik", "format": fmt}, self.hdr())
            self.assertEqual(status, 200, fmt)
        self.assertEqual(self.call("POST", "/v1/tts", {"text": "Jimlikdan keyin ham ishlaydi."}, self.hdr())[0], 200)

    def test_spoken_header_has_numbers_as_words(self):
        status, h, _ = self.call("POST", "/v1/tts", {"text": "Narxi 85 000 so‘m."}, self.hdr())
        import urllib.parse
        self.assertIn("sakson besh ming", urllib.parse.unquote(h["X-Tilsim-Spoken"]))

    def test_demo_needs_site_origin_and_short_text(self):
        site = {"Origin": "https://javohir120.github.io", "CF-Connecting-IP": "203.0.113.7"}
        status, h, _ = self.call("POST", "/v1/tts", {"text": "Sayt sinovi.", "format": "wav"}, site)
        self.assertEqual((status, h["Content-Type"]), (200, "audio/mpeg"))  # the site always gets MP3
        self.assertEqual(h["Access-Control-Allow-Origin"], "https://javohir120.github.io")
        status, _, body = self.call("POST", "/v1/tts", {"text": "a" * 301}, site)
        self.assertEqual((status, json.loads(body)["code"]), (400, "too_long"))
        status, _, _ = self.call("POST", "/v1/tts", {"text": "Salom."}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 401)

    def test_daily_demo_cap_for_the_cloud(self):
        site = {"Origin": "https://javohir120.github.io"}
        old = serve.DEMO_DAILY_MAX
        serve.DEMO_DAILY_MAX = 2
        try:
            serve.Handler.service.demo_day = ("", 0)
            codes = []
            for n in range(3):
                h = {**site, "CF-Connecting-IP": f"198.51.100.{n}"}  # different visitors
                status, _, body = self.call("POST", "/v1/tts", {"text": f"Kunlik limit {n}."}, h)
                codes.append(status if status == 200 else json.loads(body)["code"])
            self.assertEqual(codes, [200, 200, "daily_limit"])
        finally:
            serve.DEMO_DAILY_MAX = old

    def test_visitor_ip_from_proxy_header(self):
        site = {"Origin": "https://javohir120.github.io"}
        svc = serve.Handler.service
        before = set(svc.ip_hits)
        self.call("POST", "/v1/tts", {"text": "Proksi sinovi."},
                  {**site, "X-Forwarded-For": "1.1.1.1, 203.0.113.99"})
        import hashlib
        want = hashlib.sha256(b"tilsim:203.0.113.99").hexdigest()[:10]
        self.assertIn(want, set(svc.ip_hits) - before)  # the proxy-added (last) address, not the claimed one

    def test_key_sync_limit_is_600(self):
        status, _, body = self.call("POST", "/v1/tts", {"text": "a " * 301}, self.hdr())
        self.assertEqual((status, json.loads(body)["code"]), (400, "too_long"))

    def test_bad_input(self):
        cases = [({"text": ""}, "empty_text"), ({"text": "x", "format": "flac"}, "bad_format"),
                 ({"text": "x", "voice": "nobody"}, "unknown_voice"), ({"text": "x", "seed": "a"}, "bad_seed")]
        for body, code in cases:
            status, _, out = self.call("POST", "/v1/tts", body, self.hdr())
            self.assertEqual((status, json.loads(out)["code"]), (400, code), body)

    def test_bad_json_then_connection_still_fine(self):
        req = urllib.request.Request(self.base + "/v1/tts", data=b"{nope", method="POST",
                                     headers={"Content-Type": "application/json", **self.hdr()})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.call("GET", "/v1/health")[0], 200)

    def test_rate_limit(self):
        first = self.call("POST", "/v1/tts", {"text": "Birinchi so‘rov."}, self.hdr(self.slow_key))
        second = self.call("POST", "/v1/tts", {"text": "Ikkinchi so‘rov."}, self.hdr(self.slow_key))
        self.assertEqual(first[0], 200)
        self.assertEqual((second[0], json.loads(second[2])["code"]), (429, "rate_limited"))
        self.assertIn("Retry-After", second[1])

    def test_normalize(self):
        status, _, body = self.call("POST", "/v1/normalize", {"text": "2026-yil 15-noyabr. Salom!"}, self.hdr())
        out = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("ikki ming yigirma oltinchi", out["spoken"])
        self.assertEqual(len(out["sentences"]), 2)

    def test_job_flow_and_privacy(self):
        text = " ".join(f"Bu {n}-gap." for n in range(1, 8))
        status, _, body = self.call("POST", "/v1/jobs", {"text": text, "format": "ogg"}, self.hdr())
        job = json.loads(body)
        self.assertEqual(status, 202)
        self.assertGreaterEqual(job["progress"]["total"], 2)  # short sentences get merged
        for _ in range(100):
            view = json.loads(self.call("GET", job["status_url"], headers=self.hdr())[2])
            if view["status"] == "done":
                break
            time.sleep(0.05)
        self.assertEqual(view["status"], "done")
        self.assertEqual(view["progress"]["done"], view["progress"]["total"])
        status, h, audio = self.call("GET", job["audio_url"], headers=self.hdr())
        self.assertEqual((status, h["Content-Type"]), (200, "audio/ogg"))
        self.assertTrue(audio.startswith(b"OggS"))
        # another key cannot see it, and no key at all is refused
        other = self.call("GET", job["status_url"], headers=self.hdr(self.slow_key))
        self.assertEqual(other[0], 404)
        self.assertEqual(self.call("GET", job["status_url"])[0], 401)

    def test_jobs_need_a_key(self):
        site = {"Origin": "https://javohir120.github.io"}
        self.assertEqual(self.call("POST", "/v1/jobs", {"text": "Salom."}, site)[0], 401)

    def test_usage_counts(self):
        before = json.loads(self.call("GET", "/v1/usage", headers=self.hdr())[2])
        self.call("POST", "/v1/tts", {"text": "Hisob sinovi, yangi matn."}, self.hdr())
        after = json.loads(self.call("GET", "/v1/usage", headers=self.hdr())[2])
        self.assertEqual(after["requests"], before["requests"] + 1)
        self.assertIsNone(after["limits"]["monthly_chars"])


if __name__ == "__main__":
    unittest.main()
