"""normalize(): written Uzbek -> the exact words a TTS voice should say.

    >>> normalize("2026-yil 14-noyabr, narxi 50.000 so'm (12,5% chegirma)")
    "ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr, narxi ellik ming so'm, o'n ikki butun o'ndan besh foiz chegirma"

Order matters: Cyrillic -> apostrophes -> lexicon -> abbreviations -> numbers -> acronyms -> symbols -> tidy.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from .numbers import cardinal, decimal, fraction, ordinal, roman_to_int, with_suffix
from .translit import cyr_to_lat, has_cyrillic, new_to_old, normalize_apostrophes

DATA = Path(__file__).resolve().parent / "data"

MONTHS = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr"]
_MONTH_RE = "|".join(MONTHS)

LETTER_NAMES = {
    "A": "a", "B": "be", "C": "si", "D": "de", "E": "e", "F": "ef", "G": "ge", "H": "ha", "I": "i", "J": "je",
    "K": "ka", "L": "el", "M": "em", "N": "en", "O": "o", "P": "pe", "Q": "qa", "R": "er", "S": "es", "T": "te",
    "U": "u", "V": "ve", "W": "dabl-yu", "X": "xa", "Y": "ye", "Z": "ze", "O'": "o'", "G'": "g'a", "SH": "sha",
    "CH": "che",
}

# Units read only right after a number. Longest first, so «km/soat» wins over «km».
UNITS = {
    "km/soat": "kilometr soatiga", "km²": "kvadrat kilometr", "m²": "kvadrat metr", "m³": "kub metr",
    "°C": "daraja", "°": "daraja", "mlrd": "milliard", "mln": "million", "trln": "trillion",
    "km": "kilometr", "sm": "santimetr", "mm": "millimetr", "kg": "kilogramm", "mg": "milligramm",
    "ml": "millilitr", "kVt": "kilovatt", "MB": "megabayt", "GB": "gigabayt", "TB": "terabayt",
    "min": "daqiqa", "sek": "soniya", "m": "metr", "g": "gramm", "l": "litr",
    "UZS": "so'm", "USD": "dollar", "EUR": "yevro", "RUB": "rubl",
}
CURRENCY_SIGNS = {"$": "dollar", "€": "yevro", "£": "funt sterling", "₽": "rubl", "₸": "tenge"}

# Case endings that may follow a number: 5-ga, 10ga, 5ta, 3-dan ...
CASE_SUFFIXES = {
    "ga", "ka", "qa", "da", "dan", "ning", "ni", "gacha", "dagi", "dek", "ta", "tadan", "tasi", "tasini",
    "tagina", "talab", "lik", "li",
}
ROMAN_KEYWORDS = "asr|bob|qism|jild|bo'lim|sinf|kurs|chorak|tom|bosqich|daraja|son"

ABBREVIATIONS = [
    (r"\bva h\.\s?k\.", "va hokazo"),
    (r"\bmil\.\s?avv\.", "miloddan avvalgi"),
    (r"\bva b\.", "va boshqalar"),
    (r"\bprof\.", "professor"),
    (r"\bakad\.", "akademik"),
    (r"\btel\.", "telefon"),
    (r"(?<![\w'])ko'ch\.", "ko'chasi"),
    (r"(?<![\w'])sh\.", "shahri"),
]

SYMBOLS = [
    ("&", " va "), ("№", " raqam "), ("+", " plyus "), ("=", " teng "), ("@", " kuchukcha "),
    ("°", " daraja "), ("%", " foiz "), ("#", " "), ("*", " "), ("_", " "), ("/", " "), ("|", " "),
    ("~", " "), ("^", " "), ("(", ", "), (")", ", "), ("[", ", "), ("]", ", "),
]

_SEP = r"[   ]"
# 50.000 / 125 000 / 1 000 000,5  or  42 / 2,5 / 3.14
NUM = rf"(?:\d{{1,3}}(?:(?:{_SEP}|\.)\d{{3}})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
_NB = r"(?<![\d.,])"  # a number never starts in the middle of another one


# ---------------------------------------------------------------- tables

def _load_tsv(name: str) -> dict[str, str]:
    table: dict[str, str] = {}
    path = DATA / name
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("\t")
        if key.strip() and value.strip():
            table[key.strip()] = value.strip()
    return table


@lru_cache(maxsize=1)
def _lexicon() -> tuple[re.Pattern | None, dict[str, str]]:
    table = _load_tsv("lexicon.tsv")
    stems = {k[:-1]: v for k, v in table.items() if k.endswith("*")}  # «sentabr*» also covers sentabrda, sentabrga …
    table = {k: v for k, v in table.items() if not k.endswith("*")}
    for src in (table, stems):
        for key, value in list(src.items()):
            if key.islower():  # «email» also matches «Email» at a sentence start
                src.setdefault(key[0].upper() + key[1:], value[0].upper() + value[1:])
    if not table and not stems:
        return None, {}
    alts = [re.escape(k) + r"(?![\w'])" for k in sorted(table, key=len, reverse=True)]
    alts += [re.escape(k) + r"(?=[\w']*)" for k in sorted(stems, key=len, reverse=True)]
    pattern = re.compile(r"(?<![\w'])(" + "|".join(alts) + ")")
    return pattern, {**stems, **table}


@lru_cache(maxsize=1)
def _acronyms() -> dict[str, str]:
    return _load_tsv("acronyms.tsv")


def reload_tables() -> None:
    """Call after editing lexicon.tsv / acronyms.tsv in a running process."""
    _lexicon.cache_clear()
    _acronyms.cache_clear()


# ---------------------------------------------------------------- numbers

def _digits(s: str) -> str:
    return " ".join(cardinal(int(d)) for d in s)


def _parse(raw: str) -> tuple[str, str | None]:
    raw = re.sub(_SEP, "", raw)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):  # dot as thousands separator: 50.000
        whole, _, frac = raw.partition(",")
        return whole.replace(".", ""), frac or None
    m = re.fullmatch(r"(\d+)(?:[.,](\d+))?", raw)
    return (m.group(1), m.group(2)) if m else (raw, None)


def number_words(raw: str) -> str:
    whole, frac = _parse(raw)
    if frac == "5":  # 2,5 -> «ikki yarim», the way people say it
        return "yarim" if int(whole) == 0 else f"{cardinal(int(whole))} yarim"
    if frac is not None:
        return decimal(whole, frac)
    if (len(whole) > 1 and whole.startswith("0")) or len(whole) > 15:  # codes, ids: digit by digit
        return _digits(whole)
    return cardinal(int(whole))


def _phone(m: re.Match) -> str:
    plus = "plyus " if m.group(0).startswith("+") else ""
    parts = [_digits(g) if g.startswith("0") else cardinal(int(g)) for g in m.groups()]
    return f"{plus}to'qqiz yuz to'qson sakkiz, " + ", ".join(parts)


def _date(day: str, month: str, year: str) -> str | None:
    d, mo = int(day), int(month)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    return f"{ordinal(int(year))} yil {ordinal(d)} {MONTHS[mo - 1]}"


def _time(m: re.Match) -> str:
    hours, minutes, suffix = m.group(1), m.group(2), m.group(3)
    words = cardinal(int(hours))
    if minutes != "00":
        words += " " + (f"nol {cardinal(int(minutes))}" if minutes.startswith("0") else cardinal(int(minutes)))
    if suffix:
        words = with_suffix(words, suffix) if suffix in CASE_SUFFIXES else f"{words} {suffix}"
    return words


def _range(m: re.Match) -> str:
    a, b, noun, counter = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    if noun:  # 1990–1995-yillarda: both ends are ordinals
        return f"{ordinal(a)}, {ordinal(b)} {noun}"
    if counter:  # 5-6 ta -> besh-oltita
        return f"{cardinal(a)}-{with_suffix(cardinal(b), counter)}"
    return f"{with_suffix(cardinal(a), 'dan')} {with_suffix(cardinal(b), 'gacha')}"


def _hyphen_suffix(m: re.Match) -> str:
    n, suffix = int(m.group(1)), m.group(2)
    low = suffix.lower()
    if low in CASE_SUFFIXES:
        return with_suffix(cardinal(n), low)
    if low in ("inchi", "nchi", "chi"):
        return ordinal(n)
    return f"{ordinal(n)} {suffix}"


def _roman(m: re.Match) -> str:
    value = roman_to_int(m.group(1))
    return f"{ordinal(value)} " if value else m.group(0)


def _glued(m: re.Match) -> str:
    words, suffix = number_words(m.group(1)), m.group(2)
    return with_suffix(words, suffix) if suffix in CASE_SUFFIXES else f"{words} {suffix}"


def _percent(m: re.Match) -> str:
    suffix = m.group(2)
    return f"{number_words(m.group(1))} {with_suffix('foiz', suffix) if suffix else 'foiz'}"


@lru_cache(maxsize=1)
def _unit_re() -> re.Pattern:
    keys = sorted(UNITS, key=len, reverse=True)
    return re.compile(rf"{_NB}({NUM}){_SEP}?(" + "|".join(re.escape(k) for k in keys) + r")(?![\w'])")


def _numbers(text: str) -> str:
    sub = re.sub
    end = r"(?!\d|[.,/]\d)"  # the number really ends here (a full stop after it is fine)
    text = sub(r"(?<=[A-Za-z])(?=\d)", " ", text)  # mp3 -> mp 3 (lexicon entries like IPv4 are already done)
    text = sub(r"(?<!\d)\+?998[\s-]?\(?(\d{2})\)?[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})(?!\d)", _phone, text)
    text = sub(rf"(?<![\d.,/])(\d{{1,2}})[./](\d{{1,2}})[./](\d{{4}}){end}",
               lambda m: _date(*m.groups()) or m.group(0), text)
    text = sub(r"(?<![\d-])(\d{4})-(\d{2})-(\d{2})(?![\d-])",
               lambda m: _date(m.group(3), m.group(2), m.group(1)) or m.group(0), text)
    text = sub(r"(?<![\d.:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])(?:-?([a-z]+)\b)?", _time, text)
    counter = r"ta(?:dan|si|sini|gina)?(?![\w'])"  # «2 ta» is said as one word: «ikkita»
    text = sub(rf"(?<![\d.,])(\d+)(?:\s*[–—]\s*|-)(\d+){end}(?:-([a-zA-Z'][\w']*)|\s?({counter}))?", _range, text)
    text = sub(rf"{_NB}({NUM}) ({counter})", lambda m: with_suffix(number_words(m.group(1)), m.group(2)), text)
    text = sub(r"(?<![\d.,])(1\d{3}|20\d{2})\s+(yil[\w']*|y\.)",
               lambda m: f"{ordinal(int(m.group(1)))} {'yil' if m.group(2) == 'y.' else m.group(2)}", text)
    month = rf"(?:{_MONTH_RE})(?:da|dan|ga|gacha|ning|ni|dagi)?(?![\w'])"  # «mart», not «marta» (times)
    text = sub(rf"(?<![\d.,])([12]?\d|3[01])(?:-|\s+)(?={month})", lambda m: f"{ordinal(int(m.group(1)))} ", text)
    text = sub(r"(?<![\d.,])(\d+)-([a-zA-Z'][\w']*)", _hyphen_suffix, text)
    text = sub(rf"(?<![\w'])([IVXLCDM]{{1,7}})(?:-|\s+)(?=(?:{ROMAN_KEYWORDS}))", _roman, text)
    text = sub(r"(?:(?<=\s)|^)[−-](?=\d)", "minus ", text)
    text = sub(rf"{_NB}({NUM}){_SEP}?%(?:{_SEP}?-?(ga|gacha|da|dan|ni|ning|lik|li)(?![\w']))?", _percent, text)
    signs = "".join(re.escape(s) for s in CURRENCY_SIGNS)
    text = sub(rf"([{signs}]){_SEP}?({NUM})", lambda m: f"{number_words(m.group(2))} {CURRENCY_SIGNS[m.group(1)]}",
               text)
    text = sub(rf"{_NB}({NUM}){_SEP}?([{signs}])", lambda m: f"{number_words(m.group(1))} {CURRENCY_SIGNS[m.group(2)]}",
               text)
    text = _unit_re().sub(lambda m: f"{number_words(m.group(1))} {UNITS[m.group(2)]}", text)
    text = sub(r"(?<![\d/])(\d+)/(\d+)(?![\d/])", lambda m: fraction(int(m.group(1)), int(m.group(2))), text)
    text = sub(rf"{_NB}({NUM})([a-z][a-z']*)", _glued, text)
    text = sub(rf"{_NB}({NUM})", lambda m: number_words(m.group(1)), text)
    return text


# ---------------------------------------------------------------- words

def _spell(acronym: str) -> str:
    names, i = [], 0
    while i < len(acronym):
        pair = acronym[i:i + 2]
        if pair in ("SH", "CH", "O'", "G'"):
            names.append(LETTER_NAMES[pair])
            i += 2
        else:
            names.append(LETTER_NAMES.get(acronym[i], acronym[i].lower()))
            i += 1
    return " ".join(names)


_CAPS_WORD = r"(?:[A-Z]|O'|G')(?:[A-Z]|O'|G'|')+"


def _acronym(m: re.Match) -> str:
    word = m.group(1)
    table = _acronyms()
    if word in table:
        return table[word]
    if len(word.replace("'", "")) >= 5:
        return word[0] + word[1:].lower()
    return _spell(word)


def _headlines(text: str) -> str:
    """Three or more ALL-CAPS words in a row are shouting, not acronyms: write them in normal case."""
    table = _acronyms()

    def fix(m: re.Match) -> str:
        words = m.group(0).split(" ")
        out = [w if w.strip(",.!?:;") in table else w.lower() for w in words]
        out[0] = out[0][0].upper() + out[0][1:]
        return " ".join(out)

    return re.sub(rf"(?<![\w'])(?:{_CAPS_WORD}[,.!?:;]? ){{2,}}{_CAPS_WORD}(?![\w'])", fix, text)


def _tidy(text: str) -> str:
    text = re.sub(r"\s*[–—]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:…])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r",\s*([.!?;:…])", r"\1", text)
    text = re.sub(r"^[\s,;:]+|[\s,;]+$", "", text)
    text = re.sub(r"([.,!?;:])(?=[A-Za-z])", r"\1 ", text)
    return text.strip()


# ---------------------------------------------------------------- public API

def normalize(text: str, apostrophe: str = "'", lexicon: bool = True) -> str:
    """Return the spoken form of Uzbek text (Latin or Cyrillic in, Latin words out).

    apostrophe: the character used for o', g' and the tutuq in the output. The open Uzbek Chatterbox
    model wants ASCII "'" (it reads ʻ badly), so that is the default.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[   ]", " ", text)
    text = re.sub(r"[‐‑]", "-", text)
    if has_cyrillic(text):
        text = cyr_to_lat(text)
    text = new_to_old(text)  # the 2026 alphabet (Ö Ğ Ş Ç) is read the same; the model learned the 1995 spelling
    text = normalize_apostrophes(text)
    text = re.sub(r"[«»“”„\"]", "", text)
    if lexicon:
        pattern, table = _lexicon()
        if pattern is not None:
            text = pattern.sub(lambda m: table[m.group(1)], text)
    for pattern_str, spoken in ABBREVIATIONS:
        text = re.sub(pattern_str, spoken, text)
    text = _numbers(text)
    text = _headlines(text)
    text = re.sub(rf"(?<![\w'])({_CAPS_WORD})(?![\w'])", _acronym, text)
    for symbol, spoken in SYMBOLS:
        text = text.replace(symbol, spoken)
    text = _tidy(text)
    if apostrophe != "'":
        text = text.replace("'", apostrophe)
    return text


def check(spoken: str) -> list[str]:
    """Problems left in text that is about to be spoken (empty list = clean)."""
    issues = []
    if re.search(r"\d", spoken):
        issues.append("raqam qoldi (digits left): " + ", ".join(sorted(set(re.findall(r"\S*\d\S*", spoken)))))
    if has_cyrillic(spoken):
        issues.append("kirill harfi qoldi (Cyrillic left)")
    caps = re.findall(r"(?<![\w'])[A-Z]{2,}(?![\w'])", spoken)
    if caps:
        issues.append("bosh harfli so'z (ALL CAPS): " + ", ".join(sorted(set(caps))))
    foreign = [w for w in re.findall(r"[\w']+", spoken) if re.search(r"w|c(?!h)", w, re.I)]
    if foreign:
        issues.append("chet so'zi? (c/w, not Uzbek letters): " + ", ".join(sorted(set(foreign))))
    odd = set(re.findall(r"[^\w\s.,!?;:'\-…]", spoken))
    if odd:
        issues.append("belgi (symbols): " + " ".join(sorted(odd)))
    return issues


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Cut long text into chunks a TTS model reads well: whole sentences, each at most max_chars."""
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        current = ""
        for part in re.split(r"(?<=[,;:])\s+", sentence):
            while len(part) > max_chars:  # no comma to cut at: cut at the last space that fits
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            if current and len(current) + 1 + len(part) > max_chars:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            pieces.append(current)
    chunks: list[str] = []  # merge very short pieces with the next one
    for piece in pieces:
        if chunks and len(chunks[-1]) < 40 and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks
