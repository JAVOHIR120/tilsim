# Uzbek Voice Lab: roadmap

*Started 2026-11-14. Budget $0 until paying customers. It will launch as a paid online service.*

## 1. What "perfect" means here

ElevenLabs spends millions on GPUs and data, so we will not beat it at everything. We can beat it **at Uzbek**.
For ElevenLabs, Uzbek is a side language: only `eleven_v4` speaks it. In your own tests (2026-10-04), its premade voices
scored 6.5–7.5 on Uzbek lines, while Gemini Charon scored 8.5. Azure Sardor cannot say o' or g' at all.

So the goal is a measurable one, checked on a fixed set of 60 test sentences (numbers, names, o'/g', q/x/h, loanwords):

| Gate | Target |
|---|---|
| Pronunciation | Round-trip character error rate (CER) **< 3 %**, lower than ElevenLabs v4 and Azure Sardor on the same lines |
| It sounds like you | Speaker similarity **≥ 0.90** against a held-out clip of you (your own voice scores 0.966 against itself) |
| Your ears | In blind A/B listening you prefer ours over ElevenLabs v4 and Gemini Charon in most pairs |
| Speed | A 10-second line in under 5 s on the laptop GPU; real time on the free CPU server after phase 5 |

## 2. Key decision: start from an open Uzbek model, do not train from scratch

Training a TTS model from zero takes thousands of hours of audio and weeks on big GPUs, which is impossible at $0.
In 2026 several teams have already published open Uzbek models, so we take the best one and make it ours.

Because this becomes a **paid service**, only commercial licences count:

| Model | Built on | Licence | Uzbek data | Cloning | Use? |
|---|---|---|---|---|---|
| [UAzimov/Uzbek-tts-chatterbox](https://huggingface.co/UAzimov/Uzbek-tts-chatterbox) | Chatterbox Multilingual + LoRA | MIT | 111 h, permissive (USC, Common Voice, …) | yes | **Test first.** Open download (383 MB adapter); the base weights are already on this laptop |
| [aisha-org/navoiy-tts](https://huggingface.co/aisha-org/navoiy-tts) | CosyVoice2-0.5B | Apache-2.0 | not stated | yes | **Test second.** Has emotions (happy, sad, angry, calm…). 1.9 GB |
| [Tohirju/chatterbox-mtl-uzbek](https://huggingface.co/Tohirju/chatterbox-mtl-uzbek) | Chatterbox Multilingual | MIT | 119 h uzbekvoice (CC0) | yes | Test if access is granted (gated form). Reported median WER 14.8 % |
| [uzlm/sayro-tts-1.7B](https://huggingface.co/uzlm/sayro-tts-1.7B) | Qwen3-TTS 1.7B | own terms, gated | — | forbidden | No: cannot be a product base |
| [mrmuminov/f5-tts-uzbek](https://huggingface.co/mrmuminov/f5-tts-uzbek) | F5-TTS | CC-BY-NC | 14.6 h | yes | No: non-commercial |

**Data we may use commercially:** Common Voice Uzbek (CC0, ~265 h, many speakers) and the Uzbek Speech Corpus (105 h, CC-BY-4.0,
so it must be credited). FLEURS-uz is for testing only.
**We may not use** FeruzaSpeech (60 h, one speaker), which is for academic use only.

## 3. Architecture

```
 Browser: web studio (text → speech, voices, history)        Telegram bot / API keys (later)
        │ HTTPS                                                     │
 ┌──────▼─────────────────────────────────────────────────────────▼──────┐
 │ Oracle Always Free VM (ARM, 4 cores, 24 GB RAM, no GPU)               │
 │  • FastAPI: auth check, job queue, billing limits                     │
 │  • uznorm: Uzbek text front-end (ours)                                │
 │  • fast CPU voice (phase 5): small ONNX model, real time              │
 └──────┬──────────────────────────────┬─────────────────────────────────┘
        │ jobs                         │ audio files
 ┌──────▼──────────────┐      ┌────────▼───────────────────┐   ┌──────────────────────┐
 │ GPU worker          │      │ Oracle Object Storage      │   │ Supabase             │
 │ laptop RTX 3050     │      │ (20 GB free)               │   │ auth, Postgres:      │
 │ (later rented GPU)  │      └────────────────────────────┘   │ users, voices, jobs  │
 │ big model, cloning  │                                       └──────────────────────┘
 └─────────────────────┘
 Training: Kaggle notebooks (free GPU), never the server.
```

**What makes it better than the others:**
1. **`uznorm`, the Uzbek text front-end.** It is built (`uznorm/`, tested on 47 sentences). It handles numbers, ordinals, dates,
   money, %, phone numbers, Cyrillic, apostrophe variants, acronyms (BMT → «be em te») and loanwords (Wi-Fi → «vay-fay»).
   Most TTS mistakes in Uzbek are text mistakes, not voice mistakes.
2. **Your pronunciation dictionary.** `uznorm/data/lexicon.tsv` is a plain file that you correct, and every voice gets better from it.
3. **The measuring harness** (section 1), so "better" is a number, not a feeling.
4. **Long-form reading.** Text is split at sentence boundaries (`split_sentences`) and joined with natural pauses.

## 4. Free resources (limits change: check before relying on them)

| Need | Free option | Limits |
|---|---|---|
| Training GPU | Kaggle Notebooks (T4 ×2 or P100) | about 30 GPU hours a week, sessions up to 12 h |
| Backup GPU | Google Colab free (T4) | variable, disconnects |
| Inference GPU | your RTX 3050 Laptop, 4 GB | only while the laptop is on; 7.7 GB RAM means one job at a time |
| Server | Oracle Always Free Ampere A1 | 4 OCPU, 24 GB RAM, no GPU; commercial use allowed |
| Auth + database | Supabase free | 500 MB database, 1 GB files; pauses after a week without activity |
| Audio storage | Oracle Object Storage | about 20 GB always free |
| Website | the Oracle VM, or Cloudflare Pages | **not** Vercel Hobby, which forbids commercial use |

## 5. Phases

| Phase | What | Done when |
|---|---|---|
| **0. Foundation** ✅ | `uznorm` + tests, recording studio, this plan | built 2026-11-14 |
| **1a. First voice** ✅ | UAzimov's Uzbek LoRA running on the laptop (`engine/uz_tts.py`): your voice from a 10 s clip, numbers voiced through `uznorm`. 2026-11-14 results are below the table | your ears say whether it is good enough to build on |
| **1. Bake-off** (week 1) | Run the UAzimov and Navoiy models (Tohirju if access is granted) on the laptop with your 10 s reference clip. Build the 60-line test set. Score CER (Gemini transcribe, which you already use, or an open Uzbek ASR), similarity (`E:/tools/voiceclone/similarity.py`) and your blind listening | one base model beats Azure Sardor and ElevenLabs v4 on your blind test |
| **2. Your dataset** (weeks 1–3) | Record about 700 sentences ≈ 1.5 h in the studio, over 4–5 sessions of 45 min. The 100 starter sentences come first; the next 600 are picked for sound coverage from a CC0 sentence pool | ≥ 60 min clean audio, SNR ≥ 40 dB, no clipping |
| **3. Your "pro" voice** (weeks 3–4) | Fine-tune the winner on your recordings (LoRA, Kaggle T4, about 2–4 GPU hours) | similarity ≥ 0.90 and CER below the base model |
| **4. Product v1** (weeks 4–6) | FastAPI on Oracle, Supabase login, web studio (text → speech, speed, pauses, history, MP3/WAV download), GPU worker on the laptop | a friend can sign up and generate audio |
| **5. Always-on + cloning** (weeks 6–8) | Distil: the big model reads 10–20 h of text in your voice, and a small fast model (Matcha/Piper-style ONNX) learns from it to run in real time on the free CPU server. Add instant voice cloning with a consent recording | the server works with the laptop off |
| **6. Paid launch** | Payme/Click payments, plans and character quotas, terms of service, API keys, Telegram bot | first paying customer |

### First results (2026-11-14, `out/first-uz/`, 8 lines, RTX 3050)

| Measure | Result |
|---|---|
| Load time / VRAM | 7 s, peak 3.48 GB of 4 GB. It fits, but only one model at a time |
| Speed | 1.1–1.5 s of compute per 1 s of speech (one line was slower while the laptop was busy) |
| Endings | every sentence ended on its own; token counts match the audio length, so nothing was cut or ran on |
| Voice similarity (vs `ref.wav`) | 0.90–0.94, mean 0.92. Your own held-out clip scores 0.966 (the ceiling); voice conversion scored 0.95. Phase 3 (fine-tune on your recordings) should close the gap |
| Pronunciation | not measured yet: needs your listening, and later an ASR round-trip |

Setup notes: the adapter needs the **English** Chatterbox base (`t3_cfg`), its own 2,454-token tokenizer, and the alignment analyzer
switched **off**. The LoRA is folded into the weights at load, so no `peft` is needed. Settings: temperature 0.8, exaggeration 0.5,
cfg 0.5, repetition penalty 2.0 (1.2 cuts sentences short).

### Pronunciation work (2026-11-14, evening)

**The measuring kit** (`tools/eval_uz.py`, `tools/pick_best.py`, `tools/prosody.py`) scores each take on four things:
- **Pronunciation:** the Uzbek Whisper ASR (Kotib, Apache-2.0) writes down what it hears, and the result is compared letter by letter, with o'/g'/q/x/h/tutuq errors counted separately.
- **Naturalness:** UTMOS22.
- **Voice similarity.**
- **Pitch spread and speaking rate** against the owner's real speech (`data/eval/speaker_profile.json`: 148 Hz, 4.65 semitones spread, 5.3 vowels/s).

The test sets are `data/eval/test_uz.tsv` (36 sentences) and `test_loan.tsv` (16 loanword sentences).

**Findings:**
1. **Most errors were at the start of a sentence, not in the sounds themselves.**
   - Only 69% of takes had the first word right; 13% had junk words before it ("hammani voy buni qara").
   - **Cause:** the toolkit that trained the open LoRA used the first ~3 s of each clip as the voice prompt *and masked them out of the loss*. So the model never learned to begin an utterance.
2. **Fix: continue from real speech** (`--prefix-text`, `--prefix-end`, the `Prefix` class in `engine/uz_tts.py`).
   - The speech sequence starts with ~5.7 s of the owner's real speech tokens, and the text with their transcript, just as in training. Only the new tokens become audio.
   - On the 8 hardest sentences, first-take CER fell from 18.7% to 1.4%–2.7%, and the first word was right in 88–94% of takes (was 50%).
   - A "say a lead-in phrase, then cut at the pause" variant failed: cutting by pauses is unreliable.
3. **Best-of-N works.** Picking the best of 4 takes by ASR took CER from 6.2% to 1.0% on the old method. Production should generate 2–4 takes and keep the best.
4. **Loanwords:** the lexicon now says «sentyabr, oktyabr» (stem rule `sentabr*`, which also covers suffixed forms). The eval treats the two spellings as the same word.
5. **Prosody:**
   - Generated speech has a narrower pitch spread than the owner (~3.2 vs 4.65 semitones), with both methods.
   - The continuation method speaks ~25% faster than the owner (6.6–6.9 vs 5.3 vowels/s).
   - **Never slow audio down afterwards.** A rubberband 0.88 slow-down took UTMOS from 3.46 to 1.36, which matches the owner's earlier video experience.
   - Calmer generation settings (exaggeration 0.4, cfg 0.3, "p6-calm") are better on every count: CER 1.7%, rate 6.5, widest pitch spread (3.3). They are the new default candidate.

**For Phase 3 (fine-tune on the owner's voice), do NOT copy that recipe:**
- take the voice prompt from a *different* clip of the same speaker;
- compute the loss on every speech token, so the model learns how sentences begin.

### Benchmark against other Uzbek voices (2026-11-15)

**How it was measured:** the same test sentences went through the local ears and a native-ear judge.
- **Local ears:** Whisper CER, UTMOS, pitch spread and speaking rate.
- **Judge:** gemini-3.8-flash rated every clip **alone**, without knowing which voice it was, on native and natural (1-10). The prompt came from the owner's video eval.py.
- **Cost:** the owner capped judging at $0.50 (`tools/gemini_judge.py` keeps a ledger in `out/gemini_spend.json`). 104 calls cost $0.05.

| Voice (20 sentences) | native | natural | UTMOS | o' / g' right | rate (vowels/s) |
|---|---|---|---|---|---|
| Owner, real recording (4 sentences) | 8.25 | 7.25 | 3.62 | 100 / 100% | 5.5 |
| FLEURS real speakers (all female) | 8.15 | 7.05 | 2.30 | 100 / 100% | 8.4 |
| Azure Sardor (edge-tts) | 7.95 | 7.65 | 4.06 | 66 / 42% | 6.2 |
| Azure Madina (edge-tts) | 7.70 | 7.55 | 4.01 | 64 / 17% | 5.8 |
| **Ours, take 1** (prefix r01, calm) | 7.45 | 6.90 | 3.57 | 98 / 92% | ~6 |
| Ours, best-of-3 picked by ASR + prosody | 7.05 | 6.45 | 3.62 | 100 / 92% | |

**Lessons:**
- **UTMOS mostly measures recording cleanliness, not humanness.** Real FLEURS speakers on everyday mics scored 2.3; studio Azure scored 4+. Ours scores 3.62, *exactly* the owner's own clip (3.62), because the clone copies his mic too. **A cleaner reference recording is the direct way to raise it.**
- **Azure sounds polished but gets Uzbek letters wrong;** we are right on o'/g'/q/x but sound less natural. The judge's complaints about us:
  - too fast and choppy, "reading a list";
  - breaths or gasps after the last word, and abrupt endings (`trim()` now drops a sound after a ≥60 ms gap and fades out over 40 ms);
  - swallowed final consonants («nutq» → «nut»).
- **Speaking rate is the biggest lever.** The prefix sentence sets the pace:
  - prefix r03 (heldout) made speech 7.5–8.8 vowels/s and scored worse (6.2–6.6 natural);
  - prefix r01 gave 5.9 vowels/s and scored best.
- **Picking takes by the prosody penalty did not help** (r = 0.04 with the judge). UTMOS predicts the judge best (r = +0.41), so `pick_best.py` now ranks by CER, then UTMOS.
- **Judge noise is about ±0.3 on 20 clips.** Compare settings on the same sentences and seeds only.

**Settings rounds** (20 sentences, blind Gemini judge; $0.15 spent in total):
- **Temperature 0.7 is best:** 7.50 native / 7.10 natural, against 7.15 / 6.70 at 0.8. Values 0.5–0.6 are no better.
- **Other changes did not help:**
  - a slower setting (exaggeration 0.3, cfg 0.2);
  - a 10 s two-sentence prefix (CER went up to 3.2%);
  - the r03 prefix (too fast).
- **Final setting** (`data/voices/ovozim.json`): prefix r01, exaggeration 0.4, cfg 0.3, temperature 0.7, and best of 3 takes by CER then UTMOS. This gives CER 0.46% and UTMOS 3.80; the judge gives about 7.3–7.5 / 7.0–7.1.

**The owner's ear (2026-11-15, `out/solishtirish.mp3`, 7 sentences, Sardor / Madina / ours):** «o'zimiz yaxshi», ours is better than both Azure voices.
The Gemini judge rated Azure higher on "natural", but a native listener hears Azure's broken o'/g'.
The owner's verdict outranks the judge, so treat the judge as a tool for comparing our own settings, not as the final word.

### Brand and landing page (2026-11-15)

- **Name:** Tilsim, chosen by the owner. **Logo:** a talisman triangle in lapis→turquoise, a gold loop and a sound wave, hand-built in SVG at no cost.
- **Landing page** (`web/`): rebuilt in ElevenLabs' design language after the owner said the first version looked "AI-made".
  - The hero has segmented tabs (Text to speech / Voice cloning / API).
  - The voice orbs are a carousel of 8 real samples; each orb breathes with the audio level.
  - The studio card has a karaoke editor.
  - The quality charts are measured: g' 92/42/17, o' 100/66/64 for Tilsim/Sardor/Madina.
  - Other sections: use cases, pricing (still "tez orada"), FAQ, and a waitlist CTA.
- **Honesty rule for the page:** only measured numbers, no invented customers or reviews.
  The minimal-pairs card lists only pairs the v3 run said correctly (qo'l/ko'l, ot/o't, tog'/tok).
- **Before launch:**
  - the owner supplies the contact (Telegram or waitlist form);
  - pick a domain and host (Cloudflare Pages or Oracle; not Vercel Hobby, which forbids commercial use);
  - add a privacy and terms page.

### Live voice on the site (2026-11-15)

- **Setup:** the hero text box calls `engine/serve.py` on the laptop GPU through a Cloudflare quick tunnel.
  The address is published via a gist; README has the details.
- **It works only while the laptop runs `start-tilsim.bat`, plugged in.** On battery (seen at 11%), Windows capped the GPU at 30 W and speech came out ~10× slower: 60–89 s for one or two sentences instead of 4–17 s.
- **Next steps for an always-on service:**
  - an hourly rented GPU worker, or a fixed tunnel name with a domain;
  - streaming sentence by sentence, so long texts start playing sooner;
  - per-user keys once accounts exist.

### Cloud copy on Modal (2026-11-15)

- `deploy/modal_app.py` runs the same server on a T4 that scales to zero, at a fixed URL: `javohir120--tilsim-api.us-east.modal.run`.
  - It uses the Starter plan's $30/month credit.
  - Measured: a cold start (GPU wake + model load) is about 30–55 s; warm, one sentence takes about 6 s.
- API keys live in the `tilsim-data` volume (hashes only). New keys are written to both the cloud and the laptop store.
- Credit guards:
  - one container at most, idle for 90 s before scaling down;
  - 25 site demos a day;
  - the site does not health-check the cloud;
  - a workspace budget, set by the owner in the dashboard.
- The laptop stays the free first choice whenever it runs `start-tilsim.bat`.

### English mode and emotions (2026-11-15)

The founder asked for clean (not Uzbek-accented) English in his voice, and 8 emotions with a strength.
He chose American English, all 8 emotions, for every use (lessons, ads, bots, audiobooks).
He could not record emotion or English clips, so both are built from what the model can already do and are proven by measurement.

**Uzbek stays exactly as it was.** The old engine (git HEAD) and the new one were run on the same 3 texts and seeds:
- the speech tokens were identical (424 whole-sentence + 372 streamed);
- the whole-sentence audio was bit-identical (max difference 0.0).

So every Uzbek number measured before still holds for requests without the new fields.

**Laptop note:** the classic fp32 engine no longer fits next to the desktop apps on the 4 GB card: 3.84 GB used, spilling into
shared memory, 8–14 s of compute per second of speech. All lab runs therefore use half precision (bf16), the same as the
streaming engine.

**English mode, rounds 1–4** (`data/eval/test_en.tsv`, 40 sentences, seed 0, `tools/en_lab.py`):
- **Measured with:**
  - CER: English Whisper (`openai/whisper-large-v3-turbo`, MIT);
  - similarity: against `heldout.wav`;
  - accent: blind `gemini_judge --mode en` on 10 sentences per variant, 1–10 «native General American».
- **What the variants change:**
  - B0 is today's path: the Uzbek LoRA and the respelling.
  - Every E variant uses the stock Chatterbox English T3 (no LoRA, its own 704-token tokenizer, `uznorm.en_normalize`), with the S3Gen timbre from `ref.wav`.

| Variant | CER | UTMOS | Similarity | Native | Accent the judge named |
|---|---|---|---|---|---|
| B0 today | 9.19% | 3.75 | 0.898 | 2.9 | Central Asian, Russian |
| E1 stock English T3 | 1.03% | 4.16 | 0.890 | 4.9 | mostly South Asian |
| E2 + LoRA'd speaker projection | 0.66% | 4.11 | 0.891 | – | – |
| E3 + calm settings | 0.69% | 4.20 | 0.891 | – | – |
| E4 T3 prompt from an English clip | 0.73% | 4.22 | 0.894 | 4.2 | South Asian |
| E6 T3 prompt from the built-in voice | 0.81% | 4.23 | 0.895 | 5.2 | South Asian |
| **E7 + speaker embedding from the built-in voice** | 0.85% | 4.24 | **0.824** | **7.6** | **General American** (3× 10/10) |
| E8 prompt from the 4 takes judged most native | 0.77% | 4.15 | 0.893 | 4.8 | South Asian |
| E9 E8 + those takes as the reference | 0.94% | 4.10 | 0.875 | 4.3 | South Asian |
| E11 speaker embedding 30% towards the built-in voice | 0.73% | 4.25 | 0.881 | 6.0 | mixed |
| E10 50% | 0.77% | 4.24 | 0.855 | 7.4 | mostly American |
| E12 70% | 1.12% | 4.24 | 0.830 | 6.6 | mixed |

- **The founder heard an Indian accent in E1. The judge agreed.**
- **What causes the accent:**
  - It is mostly T3's speaker embedding (the VoiceEncoder vector of `ref.wav`): the model gives the accent that goes with that voice.
  - The speech prompt alone (E4, E6, E8) hardly matters.
- **Why similarity falls:** the built-in voice is lower (E7 takes: 131 Hz median, the founder: 146–151 Hz), so the embedding also moves the pitch.
- **Where it stands:** no mix meets both gates yet (native ≥ 8 and similarity ≥ 0.88).
- **Round 5 (2026-11-16):** the founder said yes to a native US male voice. CMU ARCTIC `bdl` was used (20 clips, 2 MB, «free for use for any purpose (commercial or otherwise)», copyright notice kept in `data/voices/english/CMU_ARCTIC_COPYING.txt`).
  - Its pitch is 127 Hz (the founder's is 146 Hz). Its embedding is closer to the founder's than the built-in voice's: cosine 0.68 against 0.56.

| Variant (T3 prompt + speaker embedding from bdl) | CER | UTMOS | Similarity | Native | ≥ 8 |
|---|---|---|---|---|---|
| **E13 bdl, full** | 0.82% | 4.24 | 0.840 | **8.3** | 9/10 (6 «General American») |
| E15 bdl 70% | 0.68% | 4.34 | 0.858 | 7.0 | 7/10 |
| E14 bdl 50% | 0.63% | 4.22 | 0.880 | 5.4 | 4/10 |

- **Decision (the founder, by ear, 2026-11-16): E13.** It is the first variant to pass the accent gate. It misses the similarity gate (0.84 < 0.88), and the founder accepted that.
  It ships as `profile["english"]`: `bdl_prompt.wav`, `bdl_ve.npy`, `spk_mix` 1.0, stock params (exaggeration 0.5, cfg 0.5, temperature 0.8, repetition penalty 1.2).
- **Laptop:** `tools/en_lab.py --t3-prompt none` (E5) crashed: T3Cond.to() cannot cast a missing prompt. That path is not used.

**Emotions, round A** (`data/eval/emo_seed.tsv`, text that fits the mood, per-mood settings grid, 2 seeds, 400 takes, `tools/emo_lab.py`):
- **Filter, then shortlist:**
  - clean takes: CER ≤ 2%, similarity ≥ 0.86, UTMOS ≥ 3.3;
  - then by acoustic direction (`tools/emo_pick.py`: pitch level, range, rate, loudness against the neutral takes);
  - 12 per mood went to the blind judge (`--mode emotion`, 8 choices, the intended mood not given).
- **First-label hits:** neutral 10/10, calm 12/12, surprised 11/12, serious 10/12, sympathetic 6/12, warm 6/12, excited 4/12 (6 heard as neutral), cheerful 2/12 (7 heard as excited).
- **No take was called over-acted.** Natural scores were 7–8.
- **Settings the judge liked:** warm exaggeration 0.55; serious 0.35; calm and sympathetic 0.25 with cfg 0.3; surprised 0.85; excited 1.1; cheerful 0.6.
- **Bank v1:** the best judged take per mood became its prefix clip (lab files: `out/emo/bank_v1/*.wav`, profile `out/emo/ovozim_emo1.json`; not shipped).
  The judge's best setting is the intensity 0.6 point; intensity 1 goes further in the same direction.

**Emotions, rounds B and C:** the same plain sentence in every mood, so the mood can only come from the voice.
- B1 (bank v1, 0.6): 4/32 first-label hits, only neutral right. B2 (+ the clip as voice prompt): 0/23. Both were judged only in part, because the network dropped.
- C1 (bank at 1.0): 0/35. C2 (extreme settings, e.g. exaggeration 1.6): 3/35.
- Quality falls fast: UTMOS down to 2.9, CER 3–5%.
- The voice does move. C2 «excited» went from 150 to 199 Hz and got louder, «sympathetic» got lower and slower.
- **The founder's ear agreed with the judge: «deyarli bir xil»** (almost the same).
- **The judge is not a valid voice-only emotion meter.** Real acted Kazakh emotional speech (KazEmoTTS, male) read on neutral news text was labelled «neutral/calm» 29 times out of 30.
  The judge reads emotion from the words, so from round K on the founder's ear decides.

**Emotions, round K (2026-11-16):** emotional speech from the internet, as the founder asked, under a commercial licence.
- Source: KazEmoTTS (ISSAI, CC BY 4.0). It is Kazakh, the closest open emotional corpus to Uzbek prosody: 2 male narrators, 6 moods.
- Only 144 clips were fetched, 54 MB, out of the 10.5 GB zip, by HTTP Range reads (`tools/remote_zip.py`).
- Attribution text required in the product: «Our product uses KazEmoTTS: Kazakh language Emotional Text-to-Speech, which is available under a Creative Commons Attribution 4.0 International License».
- Mood mapping: happy → cheerful, warm and excited; surprise → surprised; sad → sympathetic and calm; angry → serious.
  The clips (`models/kazemotts/prompts/`, not in git) were used as T3's voice prompt (K1) or as the prefix with a transliterated transcript (K2).
  The founder's speaker embedding and timbre stay.

- **Round K results** (5 plain sentences):

| | Voice change against neutral (147 Hz) | Quality |
|---|---|---|
| K1: Kazakh clip as the voice prompt | small; only excited (179 Hz, louder) and surprised (166 Hz) | CER 0.4–4.2%, UTMOS 3.21–3.72 |
| K2: Kazakh clip as the prefix | larger (excited 195 Hz; cheerful very fast) | CER up to 9%, UTMOS down to 2.75: the Kazakh text leaks into pronunciation |

- **What it shows:** with this model, a mood prompt moves the voice only a little, even when the prompt is real acted emotion.
  Audible, controllable emotion needs training (an emotion adapter on labelled emotional speech), not prompting.
- **Decision (the founder, 2026-11-16):**
  - Ship English now.
  - Keep the emotion field but refuse anything except «neytral» (`400 emotion_unavailable`) until a bank exists: the API is ready, the voice is not.
  - Run emotion training as its own project.
    Data: openly licensed emotional speech (KazEmoTTS, CC BY 4.0; CC BY Uzbek audiobooks and audio stories found on YouTube with the Creative Commons filter: Sokinlik Istab, REGISTON TV, BBC News O‘zbek).
    Channels that look like re-uploads are left out, because their CC mark may not be valid.

**English mode on the cloud T4** (dev run, `TILSIM_EN=1`, 2026-11-16):
- **Speed:** Uzbek short and medium 0.56 s to first audio, RTF 0.64 / 0.51; before English was loaded: 0.53 s, 0.60 / 0.49.
- **English and mixed text:** English 0.55 s, RTF 0.54–0.56. Mixed uz → en with `lang=auto`: 0.56 s, RTF 0.51, so switching adds nothing measurable.
- **VRAM:** 4.36 GB.
- **Laptop:** the second T3 overfills the 4 GB card (4.02 GB reserved) and makes even Uzbek 5–6× slower.
  English mode is therefore opt-in (`TILSIM_EN=1`): off on the laptop, on in the cloud.
- **Cost:** the two cloud benchmark runs, including one that failed on the missing English tokenizer, about $0.10.

**Shipped to the cloud (2026-11-16):**
- **Rollout:** `TILSIM_EN=1`, a dev copy tested first (`TILSIM_APP=tilsim-dev`, then stopped), then prod.
- **Live check from Uzbekistan:**
  - the old client (no new fields): first audio 1.02 s, cancel 0.28 s;
  - `lang=auto`: 0.81 s; `lang=en`: 0.80 s;
  - emotion: refused with `emotion_unavailable`, as intended.
- **Bug found and fixed on the way:** every turn's usage write (a ~2 s volume commit) ran on the speaking thread, so a turn sent right after «end» started ~2 s late.
  - Before the fix, later turns took 2.9–4.1 s to first audio; after it, 0.79–0.84 s.
  - Billing now runs in the background, with a test that a slow write never delays the next turn.
- **Modal cost of all cloud tests:** about $0.25 of the monthly credit.

**Gemini spend for this work:** $0.185 of the $0.75 the founder allowed (ledger $0.1526 → $0.3372, 379 calls).

<!-- EN-EMO -->

## 6. Selling it: what changes

- **Licences:** only the MIT/Apache rows in section 2. Credit CC-BY data (USC) on a "Credits" page. Keep a `LICENSES.md` listing every model and dataset in the product.
- **Consent for cloning:** clone a voice only with that person's recorded consent (like ElevenLabs' voice verification). Keep the inaudible Chatterbox watermark, and forbid impersonation in the terms.
- **Personal data law (check with a lawyer before launch):** voice recordings are personal data. Uzbekistan's personal-data law
  requires citizens' personal data to be stored on servers inside Uzbekistan, and the free Oracle and Supabase servers are abroad.
  This does not block building and testing. It may require a local server for customer voice data once real customers sign up.
- **Payments:** a registered business (YaTT or company) plus Payme or Click for Uzbek customers.
- **First money to spend:** an hourly rented GPU for the worker (so the service does not depend on the laptop), then a local server if the law requires it.

## 7. Risks

- **The laptop is small.** 4 GB VRAM and 7.7 GB RAM: run one model at a time, close the browser during long jobs, and never run training here.
- **Base-model quality is self-reported.** That is why phase 1 measures it before we commit.
- **Kaggle quota.** Plan training runs to fit 30 GPU hours a week; save checkpoints often.
- **Recording quality is the ceiling.** A laptop's built-in mic gives a noticeably worse voice than a USB mic in a quiet, soft room. The studio measures this on every take.

## Sources

- Open Uzbek models: [UAzimov Chatterbox LoRA](https://huggingface.co/UAzimov/Uzbek-tts-chatterbox) · [Navoiy TTS](https://huggingface.co/aisha-org/navoiy-tts) · [Tohirju chatterbox-mtl-uzbek](https://huggingface.co/Tohirju/chatterbox-mtl-uzbek) · [Sayro 1.7B](https://huggingface.co/uzlm/sayro-tts-1.7B) · [F5-TTS Uzbek](https://huggingface.co/mrmuminov/f5-tts-uzbek) · [Matcha-TTS uz](https://huggingface.co/OvozifyLabs/matcha-tts-uz-v1)
- Data: [FeruzaSpeech paper](https://arxiv.org/abs/2410.00035) (lists Common Voice uz at 265 h and USC at 105 h)
- Your earlier tests: `~/.claude/skills/video-production/references/voice-uzbek.md` (ElevenLabs v4, Charon and Sardor scores; the Chatterbox install at `E:/tools/voiceclone`)
