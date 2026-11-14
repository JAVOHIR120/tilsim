# Tilsim API

O‘zbekcha matnni asoschi ovozida audioga aylantiradi: MP3, WAV yoki OGG (Telegram ovozli xabari uchun).
Matnni o‘zingiz tayyorlashingiz shart emas: raqam, sana, narx, foiz, kirill yozuvi va yangi alifbo (Ö, Ğ, Ş, Ç) avtomatik to‘g‘ri o‘qiladi.

> **Ikkita server bor:**
> - **Bulut** (Modal, T4 GPU) doim ishlaydi. Ish bo‘lmasa uxlaydi va har oydagi $30 bepul kreditdan sarflaydi. 1–2 daqiqa tanaffusdan keyingi birinchi so‘rov modelni yuklagani uchun 30–60 soniya kutadi.
> - **Noutbuk** (`start-tilsim.bat`) bepul va tez, lekin faqat noutbuk yoqiq va zaryadga ulangan paytda ishlaydi.
>
> Sayt va Python mijoz avval noutbukni sinaydi, u javob bermasa bulutga o‘tadi.

## 1. Kalit

- **Sizning kalitingiz:** `E:\voicelab\.secrets\tilsim-api-key.txt` faylida. Bu fayl git'ga tushmaydi.
- **Uni hech kimga bermang va kodga yozmang.** Kodda `TILSIM_API_KEY` muhit o‘zgaruvchisidan o‘qing.
- **Server kalitni saqlamaydi.** `data/api_keys.json` faylida uning SHA-256 xeshi turadi, xolos. Kalitni yo‘qotsangiz, yangisini yarating.

Kalitni so‘rovning `X-API-Key: tlsm_...` sarlavhasida yuboring (yoki `Authorization: Bearer tlsm_...`).

**Bulutdagi kalitlar** (asosiysi):

```bash
set M=E:\tools\modal-venv\Scripts\modal.exe
%M% run deploy/modal_app.py::keys --cmd create --name "Telegram bot"            # -> .secrets/telegram-bot.key
%M% run deploy/modal_app.py::keys --cmd create --name "Mijoz A" --monthly-chars 200000 --per-minute 10
%M% run deploy/modal_app.py::keys --cmd list                                    # kalitlar va shu oydagi sarf
%M% run deploy/modal_app.py::keys --cmd revoke --key-id k_1a2b3c4d
%M% run deploy/modal_app.py::keys --cmd import                                  # noutbukdagi kalitlarni bulutga ko‘chirish
```

**Noutbukdagi kalitlar:**

```bash
python engine/apikeys.py create "Telegram bot" --out .secrets/telegram-bot.key
python engine/apikeys.py list
python engine/apikeys.py revoke k_1a2b3c4d
```

- **Qayta ishga tushirish shart emas:** yangi yoki o‘chirilgan kalit 30 soniya ichida kuchga kiradi.
- **Statistika alohida yuritiladi:** bulut va noutbuk sarfni har biri o‘zida hisoblaydi.
- **Bulutdagi kalitni noutbukda ishlatish uchun** uni `apikeys.py` bilan noutbukda ham yarating, yoki noutbuk kalitlarini `import` bilan bulutga ko‘chiring.

## 2. Server manzili

- **Bulut manzili doimiy:** `https://javohir120--tilsim-api.us-east.modal.run`. Ilovalaringizga shuni yozib qo‘yishingiz mumkin (`TILSIM_API_URL`).
- **Noutbuk tunnelining manzili** har ishga tushganda o‘zgaradi (`https://<so‘zlar>.trycloudflare.com`).

Ikkala manzil ham shu gistda turadi (`laptop` va `cloud` maydonlari):

```bash
curl -s https://api.github.com/gists/5801b83645819ab8d22340b5b60798f2 | python -c "import json,sys; print(json.loads(json.load(sys.stdin)['files']['tilsim-api.json']['content']))"
```

- **Python mijoz** (`clients/tilsim.py`) manzilni o‘zi topadi.
- **Shu kompyuterdan** so‘rov yuborsangiz, `http://127.0.0.1:8790` ham ishlaydi.
- **Doimiy manzil** kerak bo‘lsa, Cloudflare hisobi va domen bilan nomli tunnel ochish kerak (ROADMAP'ga qarang).

## 3. Tez boshlash

**Python** (`clients/tilsim.py` faylini loyihangizga ko‘chiring, faqat standart kutubxona ishlatiladi):

```python
from tilsim import Tilsim

t = Tilsim()                                   # kalit: TILSIM_API_KEY yoki .secrets/tilsim-api-key.txt
t.save("Assalomu alaykum! Buyurtmangiz qabul qilindi.", "javob.mp3")
t.save("Ertaga soat 9:30 da kutamiz.", "eslatma.ogg")             # Telegram ovozli xabari
t.save(open("kitob.txt", encoding="utf-8").read(), "kitob.mp3",   # uzun matn: o‘zi vazifaga o‘tadi
       on_progress=lambda done, total: print(f"{done}/{total}"))
print(t.usage())
```

**Buyruq satri:**

```bash
python clients/tilsim.py "Narxi 85 000 so‘m, chegirma 15%." -o narx.mp3
python clients/tilsim.py --file matn.txt -o matn.ogg
python clients/tilsim.py --normalize "2026-yil 20-noyabr, soat 9:30"
python clients/tilsim.py --usage
```

**curl:**

```bash
curl -X POST "$TILSIM_URL/v1/tts" -H "X-API-Key: $TILSIM_API_KEY" -H "Content-Type: application/json" \
     -d '{"text": "Assalomu alaykum!", "format": "mp3"}' -o salom.mp3
```

**JavaScript (Node 18+):**

```js
const r = await fetch(`${process.env.TILSIM_URL}/v1/tts`, {
  method: 'POST',
  headers: { 'X-API-Key': process.env.TILSIM_API_KEY, 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Assalomu alaykum!', format: 'ogg' }),
});
if (!r.ok) throw new Error((await r.json()).error);
await fs.promises.writeFile('salom.ogg', Buffer.from(await r.arrayBuffer()));
```

## 4. Manzillar

| Usul | Manzil | Kalit | Nima qiladi |
|---|---|---|---|
| GET | `/v1/health` | yo‘q | Server tirikmi, model yuklanganmi, navbat uzunligi |
| GET | `/v1/voices` | yo‘q | Ovozlar ro‘yxati (`asoschi`) |
| POST | `/v1/tts` | ha | Qisqa matn (600 belgigacha), javobda audio fayl |
| POST | `/v1/jobs` | ha | Uzun matn (5000 belgigacha), fonda bajariladigan vazifa |
| GET | `/v1/jobs/{id}` | ha | Vazifa holati va jarayoni (tayyor gaplar / jami) |
| GET | `/v1/jobs/{id}/audio` | ha | Tayyor vazifaning audiosi |
| POST | `/v1/normalize` | ha | Matn qanday o‘qilishini ko‘rsatadi (GPU ishlatilmaydi) |
| GET | `/v1/usage` | ha | Shu oydagi so‘rovlar, belgilar va audio soniyalari |

### POST /v1/tts

```json
{ "text": "Assalomu alaykum!", "voice": "asoschi", "format": "mp3", "seed": 0 }
```

| Maydon | Majburiy | Qiymat |
|---|---|---|
| `text` | ha | 1–600 belgi |
| `voice` | yo‘q | `asoschi` (standart) |
| `format` | yo‘q | `mp3` (standart, 96 kbit/s), `wav` (16-bit, 24 kHz), `ogg` (Opus 48 kbit/s, Telegram `sendVoice` uchun) |
| `seed` | yo‘q | Butun son, standart `0`. Natija yoqmasa, boshqa son bering: ovoz boshqacha o‘qiladi. |

**Javob:** audio baytlar (`Content-Type: audio/mpeg | audio/wav | audio/ogg`) va qo‘shimcha sarlavhalar:

- `X-Tilsim-Seconds`: audio uzunligi, soniyada;
- `X-Tilsim-Chars`: hisoblangan belgilar soni;
- `X-Tilsim-Spoken`: matn qanday o‘qilgani (URL-kodlangan).

Ovoz −16 LUFS balandlikka keltiriladi va unga sezilmas suv belgisi qo‘shiladi.

**Kesh:** bir xil matn, ovoz va `seed` qayta so‘ralsa, natija keshdan bir zumda keladi.
Bu boshqa formatdagi so‘rovlarga ham tegishli. Keshdan kelgan javob oylik hisobga qo‘shilmaydi.

### POST /v1/jobs (uzun matnlar)

Tana (`body`) `/v1/tts` niki bilan bir xil. Faqat matn 5000 belgigacha bo‘lishi mumkin.
Javob darhol `202` kodi bilan keladi:

```json
{ "id": "job_3f9a1c2b7d10", "status": "queued", "format": "mp3",
  "progress": { "done": 0, "total": 14 },
  "status_url": "/v1/jobs/job_3f9a1c2b7d10", "audio_url": "/v1/jobs/job_3f9a1c2b7d10/audio" }
```

1. `status_url` ni har 2–5 soniyada so‘rang. Holat `queued` → `working` → `done` (yoki `failed`) tartibida o‘zgaradi.
2. Holat `done` bo‘lgach, `audio_url` dan faylni oling.

Boshqa qoidalar:
- vazifa natijasi 1 soat saqlanadi;
- vazifani faqat uni yaratgan kalit ko‘ra oladi;
- bir kalitda bir vaqtda ko‘pi bilan 3 ta faol vazifa bo‘ladi.

### POST /v1/normalize

```json
{ "text": "Uchrashuv 2026-yil 20-noyabr, soat 9:30 da." }
→ { "spoken": "Uchrashuv ikki ming yigirma oltinchi yil yigirmanchi noyabr, soat to'qqiz o'ttiz da.",
    "sentences": ["..."] }
```

## 5. Xatolar

Xato bo‘lsa, javob JSON ko‘rinishida keladi: `{"error": "o‘zbekcha izoh", "code": "..."}`.
`429` va `503` kodlarida `Retry-After` sarlavhasi necha soniya kutish kerakligini aytadi. Python mijoz shuni o‘zi kutib, qayta urinadi.

| HTTP | `code` | Ma’nosi | Nima qilish kerak |
|---|---|---|---|
| 400 | `empty_text`, `too_long`, `bad_format`, `unknown_voice`, `bad_seed`, `bad_json` | So‘rov noto‘g‘ri | Matn yoki maydonlarni tuzating. 600 belgidan uzun matnni `/v1/jobs` ga yuboring |
| 401 | `no_key`, `bad_key` | Kalit yo‘q yoki noto‘g‘ri / o‘chirilgan | `X-API-Key` ni tekshiring |
| 402 | `quota_exceeded` | Kalitning oylik belgi limiti tugadi | Keyingi oyni kuting yoki limitni oshiring |
| 404 | `not_found`, `no_job` | Manzil yoki vazifa yo‘q (vazifa 1 soatdan eski bo‘lishi mumkin) | — |
| 409 | `not_ready` | Vazifa hali tayyor emas | `status_url` ni kutib turing |
| 429 | `rate_limited`, `too_many_jobs` | Daqiqalik limit yoki 3 ta faol vazifa | `Retry-After` soniya kuting |
| 503 | `loading`, `busy` | Model yuklanmoqda yoki navbat to‘la | Biroz kutib qayta urining |
| 500 | `failed` | Ovoz yaratishda xato | Qayta urining; takrorlansa `out/serve.log` ga qarang |

## 6. Limitlar va tezlik

| | API kalit | Sayt (kalitsiz) |
|---|---|---|
| Bitta so‘rovda | 600 belgi (`/v1/tts`), 5000 belgi (`/v1/jobs`) | 300 belgi |
| Tezlik limiti | daqiqasiga 60 (sizning kalitingiz) | tashrif buyuruvchiga 10 daqiqada 8 ta, kuniga 60 ta |
| Oylik limit | yo‘q (sizning kalitingiz) | — |
| Navbatdagi o‘rni | birinchi | ikkinchi (vazifalar uchinchi) |
| Formatlar | mp3, wav, ogg | mp3 |

**Tezlik** (RTX 3050, zaryadlovchi ulangan): bitta gap taxminan 4–6 soniya oladi, 1 daqiqalik audio esa taxminan 2–3 daqiqa.
Batareyada taxminan 10 baravar sekinroq ishlaydi.

## 7. Telegram bot misoli

```python
import os, requests
from tilsim import Tilsim

tilsim = Tilsim()
BOT = os.environ["TELEGRAM_BOT_TOKEN"]

def ovozli_javob(chat_id: int, matn: str):
    ogg = tilsim.speak(matn, format="ogg")
    requests.post(f"https://api.telegram.org/bot{BOT}/sendVoice",
                  data={"chat_id": chat_id}, files={"voice": ("javob.ogg", ogg, "audio/ogg")}, timeout=60)
```

## 8. Maslahatlar

- **Tinish belgilari ohangni boshqaradi.** Nuqta, vergul, so‘roq va undov belgilarini to‘g‘ri qo‘ying: gaplar orasida 0,25 soniyalik pauza bo‘ladi.
- **So‘z noto‘g‘ri o‘qilsa,** avval `/v1/normalize` bilan tekshiring. Keyin uni `uznorm/data/lexicon.tsv` ga qo‘shing (masalan, `Wi-Fi	vay-fay`).
- **Bir xil matnning boshqa varianti** kerak bo‘lsa, boshqa `seed` bering.
- **Uzun matnni** (kitob, maqola) har doim `/v1/jobs` orqali yuboring. Python mijozdagi `save()` buni o‘zi qiladi.
