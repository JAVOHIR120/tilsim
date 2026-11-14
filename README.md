# Uzbek Voice Lab

An Uzbek text-to-speech and voice lab, like ElevenLabs but built for Uzbek. The plan, the model choice and the phases are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## What is here now

| Part | What it does |
|---|---|
| `uznorm/` | The Uzbek text front-end: written text → the exact words to say. It handles numbers, dates, money, %, phones, Cyrillic, apostrophes, acronyms and loanwords. Standard library only. |
| `uznorm/data/lexicon.tsv` | Your pronunciation dictionary (Wi-Fi → vay-fay). Edit it freely. |
| `uznorm/data/acronyms.tsv` | Acronym exceptions (NATO → Nato). Others are spelled: BMT → be em te. |
| `studio/` | A local recording studio for building your voice dataset, with live checks on every take. |
| `engine/uz_tts.py` | Uzbek speech in any voice from a 10 s clip: Chatterbox + the open Uzbek LoRA (MIT), merged at load. |
| `models/uzbek-chatterbox-lora/` | That adapter (383 MB, SHA-256 checked) and its tokenizer. Not in git. |
| `data/scripts/starter_uz.tsv` | The first 100 sentences to record (about 12 minutes). |
| `data/eval/test_uz.tsv` | 36 fixed pronunciation test sentences: minimal pairs (ot/o't, qo'l/ko'l, sher/she'r), geminates, questions, numbers, loanwords, Cyrillic and 2026-alphabet input. |
| `docs/TALAFFUZ.md` | Uzbek pronunciation guide (in Uzbek): the 1995 and 2026 alphabets with IPA, stress, intonation, typical TTS errors and how we detect them. |
| `tools/features.py`, `tools/plot_features.py` | Spectrogram + pitch (F0) + loudness images of generated audio, for checking it by eye. |
| `tests/` | Normalizer tests: `python -m unittest discover -s tests` |
| `web/` | The **Tilsim** landing page (static HTML/CSS/JS) and `web/brand.html`, the brand sheet. See "Website" below. |

## Website (Tilsim)

The product name is **Tilsim** (talisman; the word holds «til», language). The page is static, so any static host works.

**Live:** https://javohir120.github.io/tilsim/. GitHub Pages serves the `gh-pages` branch, which holds only the contents of `web/`.
After changing the site, commit to `main`, then publish with:

```bash
git subtree push --prefix web origin gh-pages
```

```bash
python -m http.server 8780 --bind 127.0.0.1 --directory web
```

- **Design language:** calm and editorial, modelled on ElevenLabs. The canvas is warm off-white `#FDFCFC`, cards are `#F5F3F1` and text is black.
  Headings use Inter Tight 300 and body text uses Inter (both OFL). Buttons are black or white pills.
  Colour appears only in the grainy "voice orbs" and the gradient cards. The tokens sit at the top of `web/assets/css/site.css`.
- **Logo:** `web/assets/brand/tilsim-mark.svg` is the mark: a lapis→turquoise talisman triangle with a gold loop and a sound wave.
  `tilsim-logo.svg` is the lockup. `favicon.ico` and `apple-touch-icon.png` are rendered from the mark.
- **Platform logos:** `web/assets/logos/` comes from Simple Icons 16.37.0 (CC0). The marks belong to their owners.
- **Audio samples:** `web/assets/audio/*.mp3` are the owner's voice from `out/v3-final/best`, normalised to −16 LUFS. Every one has CER 0.
- **Cache:** after any CSS or JS edit, bump `?v=N` on both links in `index.html`.
- **Still to fill:** the contact for every "Erta kirish" button. The links marked `data-todo="contact"` point to `#` until the owner gives a Telegram link or a waitlist form.

## Record your voice

```bash
python studio/server.py --voice ovozim
```

Open http://127.0.0.1:8765 in Chrome or Edge, then:

1. Click **Mikrofonni yoqish** and allow the microphone.
2. Click **Jimlikni o'lchash** and stay silent for 3 s.
3. Read each sentence with these keys:
   - <kbd>Space</kbd> starts and stops recording;
   - <kbd>P</kbd> plays the take back;
   - <kbd>Enter</kbd> saves and moves on;
   - <kbd>R</kbd> records the sentence again;
   - <kbd>E</kbd> fixes the sentence text;
   - <kbd>←</kbd> <kbd>→</kbd> move between sentences.

The studio turns off echo cancellation, noise suppression and auto-gain, because the model must learn your real voice.
It checks every take for clipping, low level, noise (SNR), cut-off starts or ends and odd speaking speed.

Takes are saved as 48 kHz 16-bit mono WAV in `data/recordings/<voice>/wavs/`. `metadata.csv` holds `id|text|spoken`
(LJSpeech format), and `takes.jsonl` holds the quality numbers of every take. A re-recorded take moves the old file to `trash/`.
Training should re-run `normalize()` on the `text` column, so improvements to `uznorm` reach old recordings too.

## Make Uzbek speech

It runs in the Chatterbox venv at `E:/tools/voiceclone` (torch with CUDA, on the RTX 3050):

```bash
E:/tools/voiceclone/.venv/Scripts/python.exe engine/uz_tts.py "Assalomu alaykum!" --ref E:/tools/voiceclone/work/ref.wav
```

- **Input:** text goes through `uznorm` first, so digits are fine. Add `--file lines.txt` for many lines.
- **Similarity check:** `--check heldout.wav` prints how close each output sounds to that speaker.
- **Output:** files go to `out/<time>/` with a `manifest.json`.
- **Settings:** come from the voice profile `data/voices/ovozim.json` (used by default; `--voice none` turns it off).
  - Every sentence continues from the first sentence of the reference clip (`Prefix`), which fixes the bad sentence starts.
  - Values: temperature 0.8, exaggeration 0.4, cfg 0.3, repetition penalty 2.0, a 6 s voice prompt.
- **Best of N:** `--seeds 0,1,2`, then `tools/eval_uz.py` + `tools/pick_best.py` keep the best take of each sentence. On 52 test sentences that gave CER 0.24%, with 46 of 52 sentences perfect.

## Use the normalizer

```python
from uznorm import normalize, check, split_sentences
normalize("2026-yil 14-noyabr soat 14:30da, narxi 50.000 so'm (12,5% chegirma)")
# "ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr soat o'n to'rt o'ttizda, narxi ellik ming so'm, o'n ikki yarim foiz chegirma"
check("5 olma")  # lists problems left in text about to be spoken; empty means clean
```
