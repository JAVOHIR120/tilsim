# IELTS Buddy: Javohir imtihonchisiga toza inglizcha (2026-11-16)

Tilsim endi inglizcha gaplarni asoschi ovozida, o‘zbekcha urg‘usiz o‘qiy oladi. Hozircha bu imkoniyat faqat bulut serverida
(Modal) bor. IELTS Buddy aynan shu serverga ulangan.
**Hech narsani o‘zgartirmasangiz, hammasi avvalgidek ishlaydi.** Inglizcha qismlar o‘zbekcha talaffuz yozuvi bilan o‘qiladi.

## Nima o‘zgartiriladi: bitta maydon

WebSocket'ning `text` xabariga `"lang": "auto"` qo‘shing. Uni sessiya boshida bir marta yuborish yetarli: maydon keyingi
xabarlarda ham amal qiladi.

```json
{"type": "text", "text": "Endi Part 2 ga o‘tamiz. ", "lang": "auto"}
{"type": "text", "text": "Describe a place you visited that you would like to return to."}
{"type": "flush"}
```

- **`auto`:** har gapning tili o‘zi aniqlanadi. O‘zbekcha gaplar avvalgidek, inglizcha gaplar esa toza Amerikacha talaffuz bilan o‘qiladi.
  - Til gap chegarasida yoki «:» dan keyin almashadi: «Savol shunday: Do you work or are you a student?».
  - 3 so‘zgacha inglizcha iqtiboslar («"fluency" va "coherence"») o‘zbekcha gap ichida qoladi.
- **`en`:** butun matn inglizcha, masalan Part 1–3 savollari to‘liq inglizcha bo‘lsa.
- **`uz`:** avvalgi holatga qaytish.

`mark` hodisasi endi gap tilini ham aytadi: `{"type":"mark","text":"…","lang":"en","emotion":"neytral"}`.
Subtitrni shunga qarab ko‘rsatish mumkin.

## `src/lib/tilsim-stream.ts` uchun o‘zgarish (taklif)

`say()` ixtiyoriy uslub parametrini olsin. Maydonlar faqat berilganda yuboriladi, shuning uchun eski chaqiruvlar o‘zgarmaydi:

```ts
type TilsimStyle = { lang?: "uz" | "en" | "auto"; emotion?: string; intensity?: number };

/** A piece of text to speak; pieces may be any size. style is sent only when given and stays until changed. */
say(text: string, style: TilsimStyle = {}): boolean {
    if (!this.inTurn) this.node?.port.postMessage({ type: "drain", on: false });
    this.inTurn = true;
    this.turnLive = true;
    void resumeSoon(this.ctx, 0);
    return this.send({ type: "text", text, flush: false, ...style });
}
```

`src/hooks/useUzExaminer.ts` da sessiyaning birinchi `say` chaqiruvi:

```ts
tts.say(text, { lang: "auto" });   // keyingi say(text) chaqiruvlari o‘zgarishsiz qoladi
```

`onMark` ikkinchi argumenti ham bo‘lishi mumkin: `m.lang`, `m.emotion`. `tilsim-stream.js` buni allaqachon uzatadi.

## Raqamlar (bulut, O‘zbekistondan o‘lchangan, 2026-11-16)

- **Birinchi ovoz:** eski so‘rov 1,02 s, `auto` 0,81 s, inglizcha 0,80 s. Avval 1,19 s edi. Server endi har navbatdan keyin
  sarf hisobini fonda yozadi, shuning uchun navbatlar orasida kechikish yo‘q.
- **Sifat:**
  - inglizcha CER 0,82%;
  - ko‘r judge «native American» bahosi 8,3/10 (avval 2,9);
  - asoschi qulog‘i bilan tanlangan.

## Hali yo‘q

- **Emotsiya** (`emotion`): hozircha faqat `neytral`, boshqa qiymat `emotion_unavailable` qaytaradi. Alohida loyihada ishlanmoqda.
  Tayyor bo‘lganda `/v1/voices` ro‘yxatida paydo bo‘ladi, kodni o‘zgartirish shart emas.
- **Noutbuk serveri** (`start-tilsim.bat`): inglizcha rejim u yerda o‘chiq, chunki 4 GB VRAM yetmaydi. `lang: "en"` → `lang_unavailable`.
