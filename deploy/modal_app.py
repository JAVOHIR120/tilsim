"""Tilsim voice API on Modal: the same engine/serve.py on a cloud T4 GPU that sleeps when nobody uses it.

    set M=E:\\tools\\modal-venv\\Scripts\\modal.exe
    %M% deploy deploy/modal_app.py                                       # build + publish; prints the fixed URL
    %M% run deploy/modal_app.py::keys --cmd create --name "Telegram bot"  # new key -> .secrets/telegram-bot.key
    %M% run deploy/modal_app.py::keys --cmd list
    %M% run deploy/modal_app.py::keys --cmd revoke --key-id k_1a2b3c4d
    %M% run deploy/modal_app.py::keys --cmd import                       # copy data/api_keys.json (hashes) up

Cost control: the Starter plan includes $30 of compute a month.
  - One container at most, and it scales to zero 90 s after the last request.
    A cold start loads the model in about 30-60 s; the first request waits for it.
  - The website's free demo is capped at TILSIM_DEMO_DAILY_MAX requests a day for all visitors together.
    API keys keep their own limits (engine/apikeys.py).
  - Set a Workspace budget in the Modal dashboard (Settings -> Usage and billing): the hard stop.

The weights are fetched from Hugging Face at pinned revisions and checked against the SHA-256 of the files
this laptop runs (2026-11-15). Both repos are MIT. The owner's voice clip is uploaded from this PC.
"""
from __future__ import annotations

import json
import os
import re
import time
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP = "tilsim"
GPU = os.environ.get("TILSIM_GPU", "T4")
CODE = "/root/voicelab"
WEIGHTS = "/weights"
REF_WAV = Path(os.environ.get("TILSIM_REF_LOCAL", "E:/tools/voiceclone/work/ref.wav"))
DEMO_DAILY_MAX = os.environ.get("TILSIM_DEMO_DAILY_MAX", "25")

BASE = ("ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", {
    "t3_cfg.safetensors": "914cb1696f47527fe8852ca8f1fe1fa63cb34f76f9c715e84e067b744dd0da81",
    "s3gen.safetensors": "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e",
    "ve.safetensors": "f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c",
    "conds.pt": "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e",
})
LORA = ("UAzimov/Uzbek-tts-chatterbox", "23d08097d80512006065c5e1273e7157abd5cd1f", {
    "adapter_model.safetensors": "25f840509f5c4b684bfa3cbe4cd3a187a844c1dff41832dc7ee5057218dd8202",
    "adapter_config.json": "git:7571493ad9da206578d99db013e688945d94d5ed",
    "checkpoints/ckpt-20000-reconstructed/tokenizer.json": "git:d27fb3f2fd38ca39b7bbfbf83a13b3c617e551df",
})
# the versions engine/uz_tts.py runs with on the laptop (pip freeze, 2026-11-15)
RUNTIME = [
    "numpy==1.26.4", "transformers==5.2.0", "tokenizers==0.22.2", "huggingface_hub==1.33.0", "hf-xet==1.7.0",
    "diffusers==0.29.0", "safetensors==0.5.3", "librosa==0.11.0", "soundfile==0.14.0", "conformer==0.3.2",
    "einops==0.8.2", "omegaconf==2.4.0", "s3tokenizer==0.3.0", "resemble-perth==1.0.1", "pyloudnorm==0.2.0",
    "pykakasi==2.3.0", "spacy_pkuseg==1.0.1", "onnx==1.23.2", "scipy==1.17.1",
]


def fetch_weights() -> None:
    """Runs once while the image is built (on Modal's network, not the owner's phone hotspot)."""
    import hashlib
    import shutil

    from huggingface_hub import hf_hub_download

    def digest(path: Path, want: str) -> str:
        if want.startswith("git:"):
            data = path.read_bytes()
            return "git:" + hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    for (repo, revision, files), dest in ((BASE, f"{WEIGHTS}/base"), (LORA, f"{WEIGHTS}/lora")):
        Path(dest).mkdir(parents=True, exist_ok=True)
        for name, want in files.items():
            got_path = Path(hf_hub_download(repo, name, revision=revision, local_dir=f"/tmp/hf/{repo}"))
            got = digest(got_path, want)
            if got != want:
                raise RuntimeError(f"{repo}/{name}: checksum {got} != {want}")
            shutil.move(str(got_path), Path(dest) / Path(name).name)
            print(f"ok {repo}/{name}")
    shutil.rmtree("/tmp/hf", ignore_errors=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.6.0", "torchaudio==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(*RUNTIME)
    .pip_install("chatterbox-tts==0.1.7", extra_options="--no-deps")  # its pins are older than what we run
    .run_function(fetch_weights)
    .env({"TILSIM_BASE_DIR": f"{WEIGHTS}/base", "TILSIM_ADAPTER_DIR": f"{WEIGHTS}/lora",
          "TILSIM_REF": f"{CODE}/voice/ref.wav", "TILSIM_KEYS": "/data/api_keys.json",
          "TILSIM_DEMO_DAILY_MAX": DEMO_DAILY_MAX, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "uznorm", f"{CODE}/uznorm", ignore=["__pycache__", "*.pyc"])
    .add_local_file(ROOT / "engine" / "uz_tts.py", f"{CODE}/engine/uz_tts.py")
    .add_local_file(ROOT / "engine" / "serve.py", f"{CODE}/engine/serve.py")
    .add_local_file(ROOT / "engine" / "apikeys.py", f"{CODE}/engine/apikeys.py")
    .add_local_file(ROOT / "data" / "voices" / "ovozim.json", f"{CODE}/data/voices/ovozim.json")
    .add_local_file(REF_WAV, f"{CODE}/voice/ref.wav")
)
admin_image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    ROOT / "engine" / "apikeys.py", "/root/apikeys.py")

app = modal.App(APP)
data = modal.Volume.from_name("tilsim-data", create_if_missing=True)  # API keys + usage (hashes only)


@app.function(image=image, gpu=GPU, cpu=2.0, memory=8192, volumes={"/data": data},
              max_containers=1, scaledown_window=90, timeout=900)
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=420, label="tilsim-api")
def api():
    """Loads the model, then opens port 8000; Modal holds the first requests until then."""
    import threading

    sys.path[:0] = [f"{CODE}/engine", CODE]
    import serve

    def run():
        server = serve.build("0.0.0.0", 8000, preload=True, before_read=data.reload, after_write=data.commit)
        server.serve_forever()

    threading.Thread(target=run, daemon=True).start()


@app.function(image=admin_image, volumes={"/data": data}, timeout=120)
def keys_admin(cmd: str, name: str = "", key_id: str = "", per_minute: int = 30, monthly_chars: int = 0,
               max_chars: int = 5000, payload: str = "") -> dict | list:
    sys.path.insert(0, "/root")
    from apikeys import KeyStore, month_now

    store = KeyStore(Path("/data/api_keys.json"), before_read=data.reload, after_write=data.commit)
    if cmd == "create":
        rec, key = store.create(name, per_minute, monthly_chars or None, max_chars)
        return {"id": rec["id"], "name": rec["name"], "key": key}
    if cmd == "revoke":
        return {"revoked": store.revoke(key_id)}
    if cmd == "import":  # merge laptop key records (hashes + limits) into the cloud store
        incoming = json.loads(payload)["keys"]
        with store.lock:
            current = store._read()
            known = {k["id"] for k in current["keys"]}
            added = [k for k in incoming if k["id"] not in known]
            current["keys"].extend(added)
            store._write(current)
        return {"imported": [k["id"] for k in added]}
    month = month_now()
    return [{"id": k["id"], "name": k["name"], "active": k["active"], "per_minute": k.get("per_minute"),
             "monthly_chars": k.get("monthly_chars"), "this_month": k.get("usage", {}).get(month, {})}
            for k in store.data["keys"]]


@app.local_entrypoint()
def keys(cmd: str = "list", name: str = "", key_id: str = "", per_minute: int = 30, monthly_chars: int = 0,
         max_chars: int = 5000, out: str = ""):
    if cmd == "import":
        print(keys_admin.remote("import", payload=(ROOT / "data" / "api_keys.json").read_text(encoding="utf-8")))
        return
    if cmd == "create" and not name:
        raise SystemExit("--name kerak")
    res = keys_admin.remote(cmd, name=name, key_id=key_id, per_minute=per_minute, monthly_chars=monthly_chars,
                            max_chars=max_chars)
    # the laptop server must know the same keys (clients use the laptop when it is on)
    sys.path.insert(0, str(ROOT / "engine"))
    from apikeys import KeyStore, _hash
    local = KeyStore(ROOT / "data" / "api_keys.json")
    if cmd == "create":
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "key"
        path = Path(out) if out else ROOT / ".secrets" / f"{slug}.key"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(res["key"] + "\n", encoding="utf-8")
        with local.lock:
            d = local._read()
            d["keys"].append({"id": res["id"], "name": res["name"], "hash": _hash(res["key"]),
                              "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "active": True, "per_minute": per_minute,
                              "monthly_chars": monthly_chars or None, "max_chars": max_chars, "usage": {}})
            local._write(d)
        print(f"created {res['id']} «{res['name']}» in the cloud and on this laptop; key written to {path}")
    else:
        if cmd == "revoke":
            local.revoke(key_id)
        print(json.dumps(res, ensure_ascii=False, indent=1))
