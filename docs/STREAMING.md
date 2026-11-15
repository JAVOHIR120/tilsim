# Tilsim jonli ovoz (streaming): o‘lchovlar, qarorlar, narx va B reja

*2026-11-15. IELTS Buddy jonli imtihonchisi uchun. Raqamlar `tools/bench_stream.py` va `tools/sweep_stream.py`
bilan o‘lchangan; sifat `tools/render_stream.py` + `tools/eval_uz.py` (Whisper CER, UTMOS, ovoz o‘xshashligi) bilan.*

## Avval: hozirgi holat (o‘zgartirishdan oldin, fp32, butun matn)

| Joy | Matn | Birinchi ovoz | RTF | T3 / S3Gen | VRAM |
|---|---|---|---|---|---|
| Noutbuk RTX 3050, zaryadda | qisqa (9 so‘z) | 5,18 s | 1,45 | 3,12 / 2,02 s | 3,69 GB |
| | o‘rta (30) | 20,59 s | 1,26 | 12,32 / 8,11 s | 3,70 GB |
| | uzun (80) | 50,38 s | 1,38 | 32,53 / 17,58 s | 3,90 GB |
| Modal T4, iliq | qisqa | 4,30 s | 1,20 | 2,97 / 1,26 s | 3,71 GB |
| | o‘rta | 18,44 s | 1,19 | 12,66 / 5,54 s | 3,71 GB |
| | uzun | 40,38 s | 1,10 | 29,36 / 10,46 s | 3,90 GB |

Hozirgi `/v1/tts` butun matnni tayyorlab, keyin javob beradi. Shuning uchun birinchi ovoz = butun yaratish vaqti.

## Nima qilindi va har qadam nima berdi (noutbuk, o‘rta matn)

| Qadam | Birinchi ovoz | RTF | Izoh |
|---|---|---|---|
| Streaming, fp32 (bo‘laklarga bo‘lish) | 1,60 s | 3,03 | Vokoder har bo‘lakda 10 s namunani qayta hisoblaydi |
| + vokoder namunasi 4 s | 1,05 s | 1,51 | |
| + flow 6 qadam | 0,85 s | 1,25 | |
| + bf16 T3 | 0,87 s | 1,19 | **Tezlashmadi:** T3 hisobga emas, yadro ishga tushirishga tiqilgan |
| + **CUDA graph T3** (`fastt3.py`) | 0,60 s | 0,76 | 1 token 34,6 → 12,2 ms, logitlar asl bilan **bitma-bit bir xil** |
| + **flow bf16 + CUDA graph** (`fastflow.py`) | 0,35 s | 0,49 | 1 estimator chaqiruvi 121 → 22 ms |
| Lekin: uzun matn, 2–3 oqim (T4) | — | 1,6–4,5 | Pacer mayda bo‘laklarga bo‘lib, «o‘lim spirali»ga tushdi |
| + flow faqat oxirgi 1,2 s kontekstni ko‘radi; bo‘lak o‘zini qoplaydi; birinchi bo‘lak 16 token, keyingilari ≥ 12 | **0,55 s** | **0,50** | Bufer hech qachon bo‘shamaydi (`min_buffer` > 0) |

**Bekor qilingan yoki o‘zgartirilgan optimizatsiyalar:**
- **bf16 T3 graf'siz.** Foyda bermadi, faqat graf bilan birga ishlatiladi.
- **Vokoder namunasi 4 s.** CER'ni 0,97% → 1,67% ga yomonlashtirdi. Bunda T3 tokenlari bir xil edi, ya’ni farq faqat vokoderdan. Shuning uchun T4'da 10 s, noutbukda 6 s ishlatiladi.
- **3 tokenlik o‘ng kontekst bilan chiqarish.** Bo‘lak chegaralarida UTMOS 3,60 → 3,47 ga tushdi. Endi har kadrning o‘ngida kamida 8 token haqiqiy kontekst bor (`HOLDBACK`).
- **Kichik minimal bo‘lak (4 token).** Sekin GPU'da o‘lim spiraliga olib keldi: soniyasiga 6 bo‘lak, UTMOS 3,09. Endi bo‘lak kamida 12 token va Pacer uni chaqiruv narxiga qarab oshiradi.

**Sifat ablatsiyasi** (`test_uz.tsv`, 36 gap, seed 0, Whisper CER):

| Variant | CER | UTMOS | O‘xshashlik |
|---|---|---|---|
| Hozirgi (fp32, butun matn, 10 s namuna) | 0,97% | 3,66 | 0,893 |
| Streaming, 4 s namuna, eski bo‘laklar | 1,67% | 3,47 | 0,897 |
| Xuddi shu, flow 6 qadam | 1,54% | 3,47 | 0,895 |
| 4 s namuna, bo‘laklarsiz | 1,64% | 3,60 | 0,896 |
| 10 s namuna, mayda bo‘laklar (spiral) | 0,93% | 3,09 | 0,907 |

## Yakuniy natija: oldin va keyin

**Sozlamalar.**
- **Noutbuk:** T3 bf16 + CUDA graph, flow bf16 + graph, 6 qadam, vokoder namunasi 6 s.
- **Modal T4:** T3 fp16 + graph (T4'da bf16 yo‘q), flow fp16 + graph, 6 qadam, vokoder namunasi 10 s.
- **Ikkalasida:** birinchi bo‘lak 16 token (0,64 s ovoz), keyingilari Pacer bo‘yicha 12–40 token, chok joyida 8 tokenlik o‘ng kontekst va Hamming crossfade.

| Joy | Matn | Birinchi ovoz: oldin → keyin | RTF: oldin → keyin | VRAM keyin | Bufer bo‘shadimi |
|---|---|---|---|---|---|
| Noutbuk RTX 3050, zaryadda | qisqa (9 so‘z) | 5,18 → **0,55 s** | 1,45 → **0,54** | 2,74 GB | yo‘q |
| | o‘rta (30) | 20,59 → **0,55 s** | 1,26 → **0,50** | 2,74 GB | yo‘q |
| | uzun (80) | 50,38 → **0,56 s** | 1,38 → **0,48** | 2,74 GB | yo‘q |
| Modal T4, iliq | qisqa | 4,30 → **0,53 s** | 1,20 → **0,60** | 3,75 GB | yo‘q |
| | o‘rta | 18,44 → **0,53 s** | 1,19 → **0,49** | 3,75 GB | yo‘q |
| | uzun | 40,38 → **0,54 s** | 1,10 → **0,49** | 3,75 GB | yo‘q |
| Modal T4, 2 oqim bir vaqtda | o‘rta + uzun | — | eng yomoni **1,03** | 3,89 GB | ha, 1,2 s gacha |
| Modal T4, 3 oqim bir vaqtda | | — | eng yomoni **1,47** | 4,05 GB | ha, 7,8 s gacha |

*Birinchi ovoz: matn kelganidan birinchi audio bo‘lagi tayyor bo‘lguncha (serverda). RTF: yaratish vaqti / audio uzunligi.*

**Bulutda uchidan-uchigacha** (`wss://…/v1/stream`, O‘zbekistondan mobil internet orqali, 2026-11-15):
- `/v1/warm` sovuq holatda: 61 s (ikkala dvigatel yuklanadi);
- WebSocket ulanishi: 1,1 s;
- matnning birinchi bo‘lagidan birinchi tovushgacha: **1,19 s** (serverda 0,53 s, qolgani AQSh sharqigacha tarmoq);
- 8,8 s ovoz 5,1 s da keldi;
- `cancel` → `end`: **0,30 s**.

**Sifat** (`test_uz.tsv`, 36 gap, seed 0):

| Variant | CER | WER | UTMOS | O‘xshashlik |
|---|---|---|---|---|
| Oldin: fp32, butun matn | **0,97%** | 4,3% | 3,66 | 0,893 |
| Keyin, noutbuk (bf16) | 1,37% | 6,1% | 3,50 | 0,901 |
| Keyin, T4 (fp16) | 1,53% | 5,8% | 3,56 | 0,906 |
| Sinov: T4, T3 fp32 | 1,37% | 5,6% | 3,51 | 0,902 |

- **Chok joylari:** chertish yo‘q. Chok joyidagi eng katta sakrash oddiy qo‘shni namunalar orasidagidan kichik (nisbat 0,26–0,44; 1 dan past bo‘lsa eshitilmaydi).
- **CER farqi:** +0,4–0,6 punkt. Xatolar asosan bir xil qiyin gaplarda (minimal juftliklar). Yarim aniqlikdagi T3 t19 dagi «o‘n ikkinchi» ni har safar tushirib qoldirdi, fp32 T3 buni tuzatdi.
- **fp32 T3 jonli ishlatilmaydi:** T4'da RTF 0,91–1,14, birinchi ovoz 1,08 s, ikki oqimda RTF 1,73.

**Shuning uchun bulutda ikki dvigatel turadi** (`TILSIM_ENGINE=both`, ~7,5 GB):
- `/v1/tts` va `/v1/jobs` eskicha ishlaydi: fp32, butun gap, qayta urinishlar, CER 0,97%;
- jonli `/v1/stream` va `/v1/tts/stream` tezkor dvigatelda.
- **Noutbukda** 4 GB ga ikkalasi sig‘maydi. U yerda `/v1/tts` ham tezkor dvigatelda (CER 1,37%). Eskisini qaytarish uchun `TILSIM_ENGINE=classic`, lekin unda jonli oqim yo‘q.

**Inglizcha so‘zlar** (`test_mixed.tsv`, 20 aralash gap):

| Yo‘l | Natija | Baho |
|---|---|---|
| **1. Lug‘at + qayta yozish** (`uznorm/english.py`, «station» → «steyshn») | CER 16,9% (xom matnda 18,5%), UTMOS 3,70 (3,69), gap boshi to‘g‘ri 75% (70%), boshida ortiqcha tovush 0% (5%) | **Tanlandi.** $0, ovoz o‘zgarmaydi, oqimda ishlaydi. Talaffuz o‘zbekcha urg‘uli. |
| 2. Inglizcha qismlar uchun alohida inglizcha TTS (masalan, Kokoro-82M, Apache-2.0) | O‘lchanmadi | Gap o‘rtasida ovoz almashadi, chok eshitiladi. Har gapda ikki model, birinchi ovoz kechikadi. Litsenziya va espeak-ng'siz ishlashi alohida tekshirilishi kerak. |
| 3. Ko‘p tilli asos model (Chatterbox multilingual, MIT; o‘zbek tili yo‘q) | O‘lchanmadi | O‘zbek LoRA'sini qaytadan o‘qitish kerak, haftalab ish, natijasi noma’lum. |

- **CER bu yerda yuqori,** chunki o‘zbekcha Whisper inglizcha so‘zlarni yaxshi yozolmaydi. Raqamlar faqat 1-yo‘lni xom matn bilan solishtirish uchun.
- **Keyingi qadam:** butunlay inglizcha gaplarni (Part 2 kartochkasi) inglizcha ovozda aytish. Almashinuv gap chegarasida bo‘ladi, gap o‘rtasida emas.

**Maqsadlar bo‘yicha:**

| Maqsad | Natija |
|---|---|
| Birinchi ovoz noutbukda < 800 ms | **Bajarildi:** 0,55 s |
| Birinchi ovoz iliq T4'da < 1,2 s | **Bajarildi:** serverda 0,53 s, O‘zbekistondan 1,19 s |
| RTF < 0,6 | **Bajarildi:** 0,48–0,54 noutbuk, 0,49–0,60 T4 (qisqa matn chegarada) |
| Chok joylarida chertish va uzilish yo‘q | **Bajarildi:** bufer hech qachon bo‘shamadi, sakrash nisbati < 0,44 |
| CER hozirgidan yomon emas | **Jonli oqimda isbotlanmadi:** 1,37–1,53% va 0,97% (bitta seed). `/v1/tts` bulutda eskicha qoldi (0,97%). |
| VRAM ≤ 4 GB, noutbukda 1 oqim | **Bajarildi:** 2,74 GB |
| T4'da 2–3 parallel oqim | **Bajarilmadi:** 2 oqimda RTF 1,03, 3 oqimda 1,47 |

**Nima qilinmadi va nega:**
- **T4'da 2–3 oqim.** Har oqim T3'ni alohida yuritadi, T4 hisoblash quvvati ikkitasiga yetmaydi. Yechimlar:
  - T3 tokenlarini oqimlararo bitta batch'da yuritish: CUDA graph batch 4 yoki 6, taxminan 1–2 kun ish;
  - yoki L4 GPU: Modal'da T4'dan taxminan 1,4 baravar qimmat (yoqishdan oldin narxni qayta tekshiramiz), sizning «ha»ingiz kerak.
- **CER tengligi.** Bitta seed'da farq 0,4–0,6 punkt. 3 seed bilan qayta o‘lchash kerak (noutbukda ~1 soat). Agar farq qolsa, qiyin gaplar uchun T3'ning oxirgi qatlamlarini fp32 da qoldirish sinab ko‘riladi.
- **torch.compile.** Noutbukdagi venv'da (Windows) Triton yo‘q, shuning uchun uning o‘rniga CUDA graph ishlatildi: T3 2,8×, flow 5,5× tezlashdi. Bulutda (Linux) torch.compile sinab ko‘rilmadi.


## Server doim tayyor turishi: narx (Modal)

**Hisob asosi.** Modal'ning rasmiy narx sahifasi (2026-11-15):
- T4 sekundiga $0.000164;
- CPU yadro-sekundiga $0.0000131;
- xotira GiB-sekundiga $0.00000222.

Bizning konteyner (T4, 2 CPU, 8 GiB) sekundiga $0.000208 turadi, ya’ni **soatiga taxminan $0,75**.
Starter tarifida har oy **$30 bepul kredit** bor.

| Variant | Qanday ishlaydi | Oyiga | Izoh |
|---|---|---|---|
| **a) Doim iliq** (`min_containers=1`) | Bitta T4 kechayu kunduz yonib turadi | 730 soat × $0,75 ≈ **$547** (CPU va xotirani 1 yadro / 6 GiB ga tushirsa ≈ $500) | Sovuq start bo‘lmaydi. $30 krediti 1,7 kunga yetadi. **Yoqilmagan.** |
| **b) Faqat sessiya paytida iliq** (`/v1/warm` + `scaledown_window`) | Imtihon ekrani ochilganda `/v1/warm`. Sessiya davomida WebSocket ochiq, konteyner yonib turadi. Tugagach 90 s (yoki 300 s) kutib uxlaydi | Bitta 15 daqiqalik imtihon: sovuq start 0,7 daq + 15 daq + 1,5–5 daq kutish ≈ 17–21 daq ≈ **$0,22–0,26** | **$30 = 115–135 ta imtihon/oy.** Bir vaqtda 3 ta imtihon bitta konteynerda bo‘lsa, har biri ~3 baravar arzon. Hozir shunday sozlangan. |
| **c1) Noutbuk** | `start-tilsim.bat`, zaryadga ulangan | $0 | Faqat noutbuk yoqiq paytda. Sayt va mijoz avval noutbukni sinaydi. |
| **c2) Bulutdagi arzon GPU** (RunPod / Vast.ai community, RTX 3090/4090) | Doim yoqiq ijara GPU | ≈ $0,2–0,35/soat → **$150–250** | Kartaga to‘lov kerak. T4'dan 2–3 baravar tez, 6–8 oqim. |
| **c3) B reja: CPU modeli** (pastda) | Oracle bepul ARM serverida VITS | **$0** | O‘qitish kerak. Ovoz sifati Chatterbox'dan pastroq bo‘lishi mumkin. |

**Tavsiya:** **b + c1**.
- Imtihon boshlanishidan oldin `/v1/warm` chaqirilsin. O‘quvchi ko‘rsatmalarni o‘qiyotgan 30–60 soniyada server uyg‘onadi.
- Noutbuk yoqiq paytda u bepul ishlaydi.
- Oyiga 100 tadan ko‘p imtihon bo‘lsa, Modal Workspace budget'ini oshiring yoki c2 ga o‘ting.
- **Pulli narsa yoqilmagan:** hozirgi sozlama $30 kredit ichida qoladi, kredit tugasa konteyner ishga tushmaydi.

## B reja: CPU'da real vaqtdan tez ishlaydigan model (VITS / Piper uslubida)

**Maqsad:** GPU puli ketmasin, Oracle'ning bepul ARM serverida (2 OCPU / 12 GB) yoki istalgan CPU'da real vaqtdan tez ishlasin.

**Model va litsenziya** (faqat MIT/Apache yoki unga teng):

| Nima | Litsenziya | Ishlatsa bo‘ladimi |
|---|---|---|
| VITS (jaywalnut310/vits) | MIT | ha, asosiy tanlov |
| Piper, asl rhasspy/piper (arxivlangan, 2025) | MIT | ha, o‘qitish va ONNX eksport skriptlari bilan |
| Piper'ning yangi davomi OHF-Voice/piper1-gpl | **GPL-3.0** | yo‘q |
| piper-phonemize → espeak-ng | **GPL-3.0** | yo‘q. Kerak ham emas (pastga qarang) |
| facebook/mms-tts-uzb (o‘zbekcha VITS) | **CC-BY-NC 4.0** | yo‘q, tijoriy emas |
| LJSpeech (inglizcha boshlang‘ich checkpoint uchun) | public domain | ha |
| Common Voice uz jumlalari (matn) | CC0 | ha, o‘qitish matni uchun |

**espeak-ng shart emas.** O‘zbek lotin yozuvi deyarli fonetik: bitta harf bitta tovush. Piper ham VITS ham `phoneme_type: text` bilan harflar ustida o‘qitiladi.
- Belgilar to‘plami: harflar, `o‘`, `g‘`, `sh`, `ch`, `ng` va tutuq (`’`) bitta-bitta belgi.
- Matnni avval `uznorm` normallashtiradi: raqamlar, inglizcha qismlar va hokazo.
- Natijada GPL kod umuman kerak emas.

**Ma’lumot.** Ikki yo‘l bor, tavsiya: avval 2, keyin 1.
1. **Sizning studio yozuvlaringiz.** VITS uchun kamida 1–2 soat toza ovoz kerak. Hozir `data/recordings` bo‘sh.
   - Studiya tayyor (`studio/server.py`). 100 gap taxminan 12 daqiqa oladi.
   - 1 soat uchun ~500 gap kerak, 4–5 kun, kuniga 1 soatdan.
2. **Distillyatsiya.** Hozirgi Chatterbox ovozingizda 4–6 soat sintetik nutq yaratiladi.
   - Matn: Common Voice uz jumlalari (CC0) va o‘zimizning gaplar.
   - Har gap ASR bilan tekshiriladi: CER > 2% bo‘lsa tashlanadi.
   - Afzalligi: yozish shart emas, talaffuz allaqachon tekshirilgan.
   - Kamchiligi: Chatterbox'ning kichik xatolari ham o‘rganiladi. Shuning uchun keyin haqiqiy yozuvlar bilan fine-tune qilinadi.

**O‘qitish.**
- **Qayerda:** Kaggle'ning bepul GPU'si (T4/P100, haftasiga 30 soat). Noutbukda o‘qitilmaydi (ROADMAP qoidasi).
- **Qanday:** LJSpeech VITS checkpoint'idan fine-tune, harflar ustida, taxminan 80–100 ming qadam.
- **Davomiyligi:** T4'da ~10–14 soat, ya’ni Kaggle'ning 1–2 haftalik kvotasi. Checkpoint'lar har sessiyada saqlanadi.

**Kutilgan natija:**
- **Tezlik:** Piper "medium" Raspberry Pi 4'da real vaqtda ishlaydi. Oracle ARM'da RTF ≈ 0,05–0,2, birinchi tovush taxminan 0,1–0,3 s.
- **Narx:** GPU umuman kerak emas, oyiga $0.
- **Sifat:** UTMOS Chatterbox'dan pastroq bo‘lishi ehtimoli bor (VITS kam ifodali). Talaffuz CER ≈ 1–3%, o‘lchab ko‘ramiz.

**Taxminiy vaqt:**

| Qadam | Vaqt |
|---|---|
| Sintetik ma’lumot yaratish va filtrlash (noutbuk GPU, zaryadda) | 1–2 kun |
| Kaggle'da o‘qitish | 1–2 hafta (kvota bo‘yicha) |
| ONNX eksport, CPU server (`/v1/stream` bilan bir xil protokol), baholash | 2–3 kun |

O‘qitish faqat sizning «ha» javobingizdan keyin boshlanadi.
