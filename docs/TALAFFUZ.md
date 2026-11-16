# O'zbek talaffuzi: ovoz modeli uchun qo'llanma

*2026-11-14. Bu fayl ovoz tizimi nimaga e'tibor berishini belgilaydi: matnni qanday tayyorlash, xatoni qanday eshitish.
Noto'g'ri joyini ko'rsangiz, tuzating. Ona tilida so'zlashuvchining qulog'i bu yerda hal qiluvchi.*

## 1. Alifbo: amaldagi va yangi

- **1995-yilgi lotin alifbosi hozir amalda.** U 26 ta harf, 3 ta harf birikmasi (sh, ch, ng) va tutuq belgisidan iborat.
- **Yangi alifbo qonuni 2026-yil 21-oktyabrda imzolandi** va **2027-yil 22-apreldan** kuchga kiradi. Unga o'tish bosqichma-bosqich bo'ladi, eski yozuvdagi hujjatlar kuchida qoladi.
- **Yangi alifbo 28 ta harf va tutuqdan iborat:** Oʻ → **Ö**, Gʻ → **Ğ**, Sh → **Ş**, Ch → **Ç**. «Ng» endi alohida harf emas, oddiy n+g deb yoziladi.

| Kirill | 1995 lotin | 2026 lotin | Tovush (IPA) | Izoh |
|---|---|---|---|---|
| а | a | a | [æ] ~ [a] | q, gʻ, x yonida orqaroq [a] |
| о | o | o | [ɒ] | lablangan, ochiq orqa unli; turkcha «o» emas |
| ў | oʻ | ö | [o] ~ [ɵ] | «ot» va «oʻt» har xil so'z |
| у | u | u | [u] | |
| и | i | i | [i] | urg'usiz bo'g'inda qisqarib [ɪ] bo'ladi |
| е, э | e | e | [e] | so'z boshida «ye» deb yozilsa [je]: yer, yetti |
| қ | q | q | [q] | chuqur, til orqasi bilan; «k» emas |
| ғ | gʻ | ğ | [ʁ] | sirg'aluvchi; «g» emas |
| х | x | x | [χ] | dag'al, bo'g'izda; «h» emas |
| ҳ | h | h | [h] | yengil nafas tovushi |
| ш | sh | ş | [ʃ] | |
| ч | ch | ç | [tʃ] | |
| нг | ng | ng | [ŋ] | ming, tong |
| ж | j | j | [dʒ], o'zlashmada [ʒ] | jon [dʒ], jurnal [ʒ] |
| в | v | v | [v] ~ [w] | |
| ф | f | f | [ɸ] ~ [f] | |
| ъ | ʼ (tutuq) | ʼ | [ʔ] yoki oldingi unlini cho'zish | taʼlim [taːlim], sheʼr ≠ sher |

**Bizning tizimda:** foydalanuvchi qaysi yozuvda yozmasin (kirill, 1995 lotin yoki 2026 lotin), `uznorm` matnni modelga 1995 lotin
shaklida beradi. Apostrof oddiy ASCII `'` bo'ladi, chunki model shu ko'rinishda o'rgatilgan. Yangi alifboga qaytarish uchun
`old_to_new()` bor.

## 2. Ovoz modeli adashadigan joylar

| Xato | Misol (to'g'ri / xato) | Qanday tekshiramiz |
|---|---|---|
| o va oʻ farqi yo'qoladi | ot (ism) / oʻt (olov) · qol / qoʻl · soz / soʻz | juft so'zli gaplar; ASR matni |
| gʻ «g» bo'lib qoladi | gʻor, togʻ, bogʻ | ASR matni; spektrogramma (gʻ shovqinli, g portlovchi) |
| q «k» bo'lib qoladi | qoʻl (qo'l) / koʻl (suv havzasi) | juft so'zli gap |
| x «h» bo'lib qoladi | xol (yuzdagi) / hol (ahvol) | juft so'zli gap |
| tutuq tushib qoladi | sheʼr (she'r) / sher (hayvon) · taʼm / tam | juft so'zli gap |
| qo'sh undosh qisqaradi | katta, ikki, sakkiz, toʻqqiz, ammo | gap davomiyligi, quloq |
| raqam inglizcha o'qiladi | «1991» → «nineteen ninety-one» | `uznorm` har doim raqamni so'zga aylantiradi |
| turkcha yoki ozarbayjoncha urg'u | «-mı», «ü/ö» tovushlari, unlilar uyg'unligi | ASR'da turkcha so'z chiqsa, rad etiladi |
| ruscha urg'u | unlilarni yutish, «a» → [ə] | quloq |
| urg'u noto'g'ri bo'g'inda | kitób → kítob | quloq, ohang chizig'i |
| so'roq ohangi yo'q | «Kelasizmi?» oxiri ko'tarilmaydi | ohang (F0) chizig'i |

### O'zlashma so'zlar: yozilishi boshqa, aytilishi boshqa

| Yoziladi | Aytiladi | Bizda |
|---|---|---|
| sentabr, sentabrda | sentyabr, sentyabrda | lug'atda `sentabr*` → sentyabr |
| oktabr, oktabrga | oktyabr, oktyabrga | lug'atda `oktabr*` → oktyabr |

Egasi 2026-11-14 da «oktabr, sentabr» so'zlarini belgilagan edi: model ularni «sentabirga» deb, orasiga unli qo'shib o'qigan.
Qulog'i bilan tanlab, «sentyabr» va «oktyabr» variantini tasdiqladi; «noyabr» va «dekabr» esa yozilishicha qoldi.

### Faqat model uchun: so'z boshidagi «ya»

Hozirgi model so'z boshidagi «ya» dagi «a» ni «e» ga yaqinlashtiradi («yengi», «yexshi»).
Egasi quloq bilan tanlab, «yaa» deb yozilganda to'g'ri chiqishini tasdiqladi:
- **2026-11-14:** yangi, yaxshi;
- **2026-11-15:** yana, yaqin, yarim, yashil, yagona, yakshanba, yashaydi, yaratdi. Hammasida «yaa» yaxshi chiqdi.

Bu imlo qoidasi emas, faqat modelga beriladigan matn uchun hiyla. U `uznorm/respell.py` (`for_model`) ichida yashaydi,
`normalize()` da emas. Shuning uchun studiya va o'qitish matnlari to'g'ri imloda qoladi.

So'z ichidagi «ya» (noyabr) va tutuqli «ya'ni» ga tegilmaydi. Model sizning ovozingizda qayta o'qitilgach, bu qoidani yana quloq bilan tekshirish kerak.
Yangi shunday so'z topilsa, `uznorm/data/lexicon.tsv` ga qo'shiladi. Oxiri `*` bilan tugagan kalit so'z o'zagi hisoblanadi
va qo'shimchali shakllarni ham qamraydi.

## 3. Urg'u va ohang

- **Urg'u odatda so'zning oxirgi bo'g'inida bo'ladi:** kitób, kitoblár, kitoblarimíz. Qo'shimcha qo'shilsa, urg'u u bilan birga oxiriga ko'chadi.
- **Ba'zi qo'shimchalar urg'u olmaydi,** urg'u ulardan oldingi bo'g'inda qoladi:
  - so'roq **-mi**: kelasízmi;
  - kesim qo'shimchalari **-man, -san, -miz, -siz, -dir**: talabáman;
  - inkor **-ma**: bórma, bórmang;
  - yuklamalar **-chi, -ku, -da, -u/-yu**.
- **Istisno so'zlar:** ba'zi ravishlarda urg'u boshda bo'ladi (hózir, áyniqsa). Ruschadan o'zlashgan so'zlar asl urg'usini saqlaydi.
- **Ohang:**
  - «-mi»li so'roq gapda ohang oxirida ko'tariladi;
  - «nima, qachon, qayerda» so'roq gapida urg'u so'roq so'zga tushadi, oxiri pasayadi;
  - darak gapda oxiri pasayadi, undov gapda ohang kuchliroq.

## 4. Tovush qonunlari (tabiiy nutqda)

- **So'z oxirida jarangsizlanish:** kitob [kitɒp], ozod [ɒzɒt]. Ovoz shunday aytsa, bu xato emas, tabiiy talaffuz.
- **Qo'sh undosh cho'ziq aytiladi:** kat-ta, ik-ki, sak-kiz. Qisqa aytilsa, «kata» bo'lib eshitiladi.
- **Tutuq:** oldingi unlini cho'zadi (maʼno [maːno]) yoki bo'g'inni uzadi (sanʼat [sanʔat]).
- **Urg'usiz «i» va «u»** jarangsiz undoshlar orasida qisqaradi: ishtirok, kitobchi.
- **«Ng» [ŋ]** bitta tovush: ming, tong, keling.

## 5. Biz qanday "eshitamiz"

1. **ASR orqali qayta yozish.** O'zbekcha nutqni taniydigan model ovozni yana matnga aylantiradi, keyin harfma-harf solishtiriladi (CER).
   - Juft so'zlardagi xatolar alohida sanaladi: o/oʻ, g/gʻ, q/k, x/h, tutuq.
   - Whisper turidagi model xato talaffuzni o'zi "to'g'rilab" yozishi mumkin. Shuning uchun imkon bo'lsa, harfma-harf eshitadigan CTC modeli ham ishlatiladi.
2. **Tabiiylik bahosi (MOS).** Avtomatik baholovchi 1–5 ball beradi. U ingliz tilida o'rgatilgan, shuning uchun faqat nisbiy solishtirishga yaraydi.
3. **Ovoz o'xshashligi:** Chatterbox speaker embedding'i bo'yicha hisoblanadi.
4. **Spektrogramma va ohang chizig'i (F0):**
   - qirsillash, shovqin va bo'g'ilgan joylar;
   - so'roq gapda ohang ko'tarilishi;
   - pauzalar.
5. **Sizning qulog'ingiz.** Oxirgi so'z sizda: ko'r-ko'rona A/B tinglash.

Sinov gaplari: [`data/eval/test_uz.tsv`](../data/eval/test_uz.tsv). Ularda juft so'zlar, qo'sh undoshlar, so'roq va undov gaplar, raqamlar,
o'zlashma so'zlar, uzun hikoya, kirill va yangi alifbodagi yozuvlar bor.

## 6. Inglizcha rejim (`lang: "en"`)

O‘zbekcha model inglizchani o‘zbekcha urg‘u bilan o‘qiydi, chunki LoRA unga o‘zbek tovushlarini o‘rgatgan.
Shuning uchun inglizcha gaplar boshqa yo‘l bilan aytiladi:
- **Model:** Chatterbox'ning o‘zbekcha LoRA'siz inglizcha modeli, o‘zining 704 belgili tokenizeri bilan.
- **Matn:** `uznorm/en_normalize.py` raqamlarni Amerikacha o‘qiydi: «1998» → «nineteen ninety-eight».
- **Urg‘u:** asosan speaker embedding'dan keladi, ya’ni modelga berilgan «ovoz pasporti»dan.
  - Asoschining o‘zbekcha nutqidan olingan pasport bilan judge «South Asian» urg‘u eshitdi: 4,9/10.
  - Amerikalik erkak ovozining pasporti bilan (CMU ARCTIC `bdl`) natija 8,3/10, ko‘pchilik gapda «General American».
  - Tembr baribir asoschiniki: uni S3Gen asoschining yozuvidan oladi.
- **Qiyin tovushlar** (th, w/v, r, so‘z oxiridagi undoshlar): inglizcha Whisper CER 0,82% (40 gap). Avvalgi yo‘lda 9,19% edi.

## Manbalar

- Yangi alifbo:
  - [Gazeta.uz, 2026-11-05](https://www.gazeta.uz/oz/2026/11/05/alifbo-suhbat/)
  - [Dunyo AA](https://dunyo.info/kommentarii/v-uzbekskom-alfavite-poyavyatsya-bukvy-i)
  - [Daryo, 2026-07-07](https://daryo.uz/2026/07/07/lotin-yozuviga-asoslangan-ozbek-alifbosi-ozgaradi-unda-harf-birikmalari-bolmaydi/)
  - Qonun matni uchun [lex.uz](https://lex.uz) ni tekshirish kerak.
- Tovushlar (IPA) va urg'u: o'zbek fonetikasining standart tavsifi. Ona tilida so'zlashuvchi tekshirib chiqishi kerak.
