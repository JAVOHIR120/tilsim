"""Tests for the Uzbek text front-end.  Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uznorm import (cardinal, check, cyr_to_lat, decimal, fraction, new_to_old, normalize,  # noqa: E402
                    normalize_apostrophes, old_to_new, ordinal, roman_to_int, split_sentences, with_suffix)


class Numbers(unittest.TestCase):
    def test_cardinal(self):
        cases = {
            0: "nol", 7: "yetti", 10: "o'n", 11: "o'n bir", 21: "yigirma bir", 40: "qirq", 99: "to'qson to'qqiz",
            100: "yuz", 101: "yuz bir", 110: "yuz o'n", 999: "to'qqiz yuz to'qson to'qqiz", 1000: "ming",
            1001: "ming bir", 2026: "ikki ming yigirma olti", 10_000: "o'n ming", 100_000: "yuz ming",
            125_000: "yuz yigirma besh ming", 1_000_000: "bir million",
            1_250_000: "bir million ikki yuz ellik ming", 1_000_000_000: "bir milliard", -5: "minus besh",
        }
        for n, words in cases.items():
            self.assertEqual(cardinal(n), words, n)

    def test_ordinal(self):
        cases = {
            1: "birinchi", 2: "ikkinchi", 3: "uchinchi", 4: "to'rtinchi", 6: "oltinchi", 7: "yettinchi",
            9: "to'qqizinchi", 10: "o'ninchi", 20: "yigirmanchi", 40: "qirqinchi", 50: "elliginchi",
            100: "yuzinchi", 1000: "minginchi", 2026: "ikki ming yigirma oltinchi",
        }
        for n, words in cases.items():
            self.assertEqual(ordinal(n), words, n)

    def test_suffix_decimal_fraction_roman(self):
        self.assertEqual(with_suffix("besh", "ga"), "beshga")
        self.assertEqual(with_suffix("ellik", "ga"), "ellikka")
        self.assertEqual(with_suffix("qirq", "gacha"), "qirqqacha")
        self.assertEqual(decimal("2", "5"), "ikki butun o'ndan besh")
        self.assertEqual(decimal("0", "05"), "nol butun yuzdan besh")
        self.assertEqual(fraction(3, 4), "to'rtdan uch")
        self.assertEqual(roman_to_int("XXI"), 21)
        self.assertEqual(roman_to_int("MCMXCIV"), 1994)
        self.assertIsNone(roman_to_int("IIII"))
        self.assertIsNone(roman_to_int("ABC"))


class Script(unittest.TestCase):
    def test_cyrillic(self):
        cases = {
            "Ўзбекистон Республикаси": "O'zbekiston Respublikasi", "Ер юзи": "Yer yuzi", "поезд": "poyezd",
            "концерт": "konsert", "милиция": "militsiya", "маъно": "ma'no", "Шаҳар": "Shahar", "ТОШ": "TOSH",
            "ғалаба": "g'alaba", "қишлоқ": "qishloq", "компьютер": "kompyuter", "кейин": "keyin",
        }
        for cyr, lat in cases.items():
            self.assertEqual(cyr_to_lat(cyr), lat, cyr)

    def test_new_alphabet(self):
        self.assertEqual(new_to_old("Özbekiston gözal, şahar, çoy, toğ va boğ"), "O'zbekiston go'zal, shahar, choy, tog' va bog'")
        self.assertEqual(new_to_old("ŞAHAR va Şahar, ÇOY"), "SHAHAR va Shahar, CHOY")
        self.assertEqual(new_to_old("ōzbek ḡoyasi"), "o'zbek g'oyasi")  # 2019-2021 draft letters
        self.assertEqual(old_to_new("O'zbekiston go'zal shahar, choy ichamiz"), "Özbekiston gözal şahar, çoy içamiz")
        self.assertEqual(old_to_new("Is'hoq tong"), "Is'hoq tong")
        self.assertEqual(normalize("Öz ğoyalarimizni çet tilida ham aytamiz, şaharimiz 2026-yilda gözal."),
                         "O'z g'oyalarimizni chet tilida ham aytamiz, shaharimiz ikki ming yigirma oltinchi yilda go'zal.")
        self.assertEqual(normalize("ŞAHAR"), "Shahar")

    def test_apostrophes(self):
        self.assertEqual(normalize_apostrophes("oʻzbek gʻoyasi"), "o'zbek g'oyasi")
        self.assertEqual(normalize_apostrophes("san’at va ma`no"), "san'at va ma'no")
        self.assertEqual(normalize_apostrophes("tog‘ va bog‘"), "tog' va bog'")
        self.assertEqual(normalize_apostrophes("‘Salom’ dedi"), "Salom dedi")


class Normalize(unittest.TestCase):
    cases = {
        "2026-yil 14-noyabr": "ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr",
        "14.11.2026": "ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr",
        "Sana: 2026-11-14.": "Sana: ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr.",
        "2026 yilda": "ikki ming yigirma oltinchi yilda",
        "5 yil davomida": "besh yil davomida",
        "5 marta": "besh marta",
        "3 martda": "uchinchi martda",
        "Narxi 50.000 so'm": "Narxi ellik ming so'm",
        "Narxi 125 000 so'm": "Narxi yuz yigirma besh ming so'm",
        "1 000 000": "bir million",
        "2,5 kg un": "ikki yarim kilogramm un",
        "0,5 litr": "yarim litr",
        "3.14": "uch butun yuzdan o'n to'rt",
        "1,25 mln": "bir butun yuzdan yigirma besh million",
        "Inflyatsiya 12,5% ga tushdi": "Inflyatsiya o'n ikki yarim foizga tushdi",
        "25%": "yigirma besh foiz",
        "$100": "yuz dollar",
        "100 $": "yuz dollar",
        "soat 14:30da": "soat o'n to'rt o'ttizda",
        "09:05": "to'qqiz nol besh",
        "10-15 yoshdagi bolalar": "o'ndan o'n beshgacha yoshdagi bolalar",
        "Yoshi 10-15.": "Yoshi o'ndan o'n beshgacha.",
        "5-6 ta olma": "besh-oltita olma",
        "2 ta gap": "ikkita gap",
        "Har biriga 3 tadan": "Har biriga uchtadan",
        "1990–1995-yillarda": "ming to'qqiz yuz to'qsoninchi, ming to'qqiz yuz to'qson beshinchi yillarda",
        "XXI asr": "yigirma birinchi asr",
        "5-sinf": "beshinchi sinf",
        "+998 90 123 45 67": "plyus to'qqiz yuz to'qson sakkiz, to'qson, yuz yigirma uch, qirq besh, oltmish yetti",
        "5ta olma": "beshta olma",
        "50ga bo'linadi": "ellikka bo'linadi",
        "3/4 qismi": "to'rtdan uch qismi",
        "Harorat -5°C": "Harorat minus besh daraja",
        "007": "nol nol yetti",
        "350 km masofa": "uch yuz ellik kilometr masofa",
        "BMT va AQSH": "be em te va a qa sha",
        "MCHJ": "em che je",
        "ЮНЕСКО": "Yunesko",
        "IELTS imtihoni": "ayelts imtihoni",
        "Wi-Fi paroli": "vay-fay paroli",
        "Email yuboring": "Imeyl yuboring",
        "it va mushuk": "it va mushuk",  # «it» is Uzbek for dog, not IT
        "Oktabr oyi sentabrga qaraganda sovuq": "Oktyabr oyi sentyabrga qaraganda sovuq",
        "1-sentabrda": "birinchi sentyabrda",
        "Yangi yil, hammasi yaxshi": "Yangi yil, hammasi yaxshi",  # real spelling; model tricks are in for_model()
        "noyabr va dekabr": "noyabr va dekabr",
        "“Salom”, dedi u.": "Salom, dedi u.",
        "Toshkent sh., Navoiy ko'ch. 5-uy": "Toshkent shahri, Navoiy ko'chasi beshinchi uy",
        "olma, nok va h.k.": "olma, nok va hokazo",
        "SALOM DO'STLAR VA MEHMONLAR": "Salom do'stlar va mehmonlar",
        "Navoiy (1441–1501)": "Navoiy, ming to'rt yuz qirq birdan ming besh yuz birgacha",
    }

    def test_cases(self):
        for written, spoken in self.cases.items():
            self.assertEqual(normalize(written), spoken, written)

    def test_apostrophe_option(self):
        self.assertEqual(normalize("o'n to'rt", apostrophe="ʻ"), "oʻn toʻrt")

    def test_output_is_clean(self):
        for written in self.cases:
            self.assertEqual(check(normalize(written)), [], written)

    def test_check_flags(self):
        self.assertTrue(check("5 olma"))
        self.assertTrue(check("BMT"))
        self.assertTrue(check("Ер"))


class ForModel(unittest.TestCase):
    def test_initial_ya(self):
        from uznorm import for_model
        self.assertEqual(for_model("Yangi yil yaqin, hammasi yaxshi"), "Yaangi yil yaaqin, hammasi yaaxshi")
        self.assertEqual(for_model("Yakshanba kuni yashil bog'da"), "Yaakshanba kuni yaashil bog'da")
        self.assertEqual(for_model("noyabr, dunyo, qiyin, ya'ni"), "noyabr, dunyo, qiyin, ya'ni")
        self.assertEqual(for_model("U yer, yil, yo'l"), "U yer, yil, yo'l")


class Split(unittest.TestCase):
    def test_split(self):
        text = ("Bu birinchi gap. " * 3 + "Bu juda uzun gap, unda vergullar bor, ular orqali bo'linadi, "
                "chunki u ikki yuz yigirma belgidan oshib ketadi, shuning uchun kesiladi, va yana davom etadi, "
                "oxirigacha bir necha so'z qo'shamiz, toki uzunligi yetarli bo'lmaguncha davom etamiz.")
        chunks = split_sentences(text, max_chars=120)
        self.assertTrue(all(len(c) <= 120 for c in chunks), chunks)
        self.assertEqual(" ".join(chunks).split(), text.split())


class EnglishRespellTest(unittest.TestCase):
    """uznorm/english.py: English stretches of mixed text are respelled, Uzbek text never changes."""

    def setUp(self):
        from uznorm.english import respell_english
        self.r = respell_english

    def test_examiner_sentence(self):
        self.assertEqual(self.r("Endi Part ikki ga o'tamiz. Describe a place you like to visit."),
                         "Endi Part ikki ga o'tamiz. Diskrayb e pleys yu layk tu vizit.")

    def test_two_language_words_stay_uzbek_alone(self):
        for t in ["Bu yerda band emas.", "Men test topshirdim.", "U it boqadi.", "Yanvar, may va iyun.",
                  "Bunday okean real emas, qanday teatr?", "Kutubxonada qulay xona bor."]:
            self.assertEqual(self.r(t), t)

    def test_single_clear_english_word_inside_uzbek(self):
        self.assertEqual(self.r("ayelts imtihonida pronunciation ham baholanadi."),
                         "ayelts imtihonida pronansieyshn ham baholanadi.")

    def test_every_uzbek_test_sentence_is_untouched(self):
        root = Path(__file__).resolve().parents[1]
        for f in ("data/eval/test_all.tsv", "data/scripts/starter_uz.tsv"):
            for ln in (root / f).read_text(encoding="utf-8").splitlines():
                if ln.strip() and not ln.startswith("#"):
                    n = normalize(ln.split("\t")[-1])
                    self.assertEqual(self.r(n), n, f"{f}: {n}")

    def test_unknown_english_word_by_rules(self):
        self.assertEqual(self.r("Please describe the weather station."), "Pliz diskrayb di veder steyshn.")


if __name__ == "__main__":
    unittest.main()
