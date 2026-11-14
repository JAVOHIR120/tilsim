"""Uzbek Cyrillic -> Latin (the official 1995 alphabet), and apostrophe clean-up."""
from __future__ import annotations

import re

_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "x", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "'", "ь": "", "ы": "i", "э": "e", "ю": "yu", "я": "ya",
    "ё": "yo", "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h",
}
_CYR_VOWELS = set("аеёиоуэюяўы")
_CYR_RE = re.compile(r"[Ѐ-ӿ]")

# Every character people type for the o' / g' mark and the tutuq (glottal stop).
APOSTROPHES = "'‘’ʻʼ`´′"
_APOS_CLASS = "[" + re.escape(APOSTROPHES) + "]"


def has_cyrillic(text: str) -> bool:
    return bool(_CYR_RE.search(text))


def cyr_to_lat(text: str) -> str:
    """'Ўзбекистон' -> "O'zbekiston"; 'ер' -> 'yer'; 'концерт' -> 'konsert'; 'маъно' -> "ma'no"."""
    out: list[str] = []
    for i, ch in enumerate(text):
        low = ch.lower()
        if low not in _MAP and low not in "ец":
            out.append(ch)
            continue
        prev = text[i - 1].lower() if i else ""
        if low == "е":
            starts = not prev or not prev.isalpha() or prev in _CYR_VOWELS or prev in "ъь"
            lat = "ye" if starts else "e"
        elif low == "ц":
            lat = "ts" if prev in _CYR_VOWELS else "s"
        else:
            lat = _MAP[low]
        if ch.isupper() and lat:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            prv = text[i - 1] if i else ""
            whole_word_upper = nxt.isupper() or (prv.isupper() and not nxt.isalpha())
            lat = lat.upper() if whole_word_upper else lat[0].upper() + lat[1:]
        out.append(lat)
    return "".join(out)


_NEW_TO_OLD = {
    # 2026 alphabet law (OʻRQ, signed 2026-10-21, in force 2027-04-22): Ö Ğ Ş Ç replace Oʻ Gʻ Sh Ch, «ng» is just n+g
    "ö": "o'", "ğ": "g'", "ş": "sh", "ç": "ch",
    # letters from the 2019-2021 drafts and look-alikes people type
    "ō": "o'", "ḡ": "g'", "ș": "sh", "ŏ": "o'", "ǧ": "g'",
}
_NEW_RE = re.compile("[" + "".join(_NEW_TO_OLD) + "".join(k.upper() for k in _NEW_TO_OLD) + "]")


def new_to_old(text: str) -> str:
    """New-alphabet text -> the 1995 Latin the TTS model was trained on: 'Özbekiston, şahar, çoy' -> "O'zbekiston, shahar, choy"."""
    def swap(m: re.Match) -> str:
        ch = m.group(0)
        old = _NEW_TO_OLD[ch.lower()]
        if not ch.isupper():
            return old
        i = m.start()
        nxt = text[i + 1] if i + 1 < len(text) else ""
        prv = text[i - 1] if i else ""
        whole_word_upper = nxt.isupper() or (prv.isupper() and not nxt.isalpha())
        return old.upper() if whole_word_upper else old[0].upper() + old[1:]

    return _NEW_RE.sub(swap, text)


def old_to_new(text: str) -> str:
    """1995 Latin -> 2026 alphabet, for display: "O'zbekiston go'zal, choy ichamiz" -> 'Özbekiston gözal, çoy içamiz'.

    Expects normalized apostrophes. «s'h» / «c'h» (tutuq between) stay two sounds, as in «Is'hoq».
    """
    pairs = [("O'", "Ö"), ("o'", "ö"), ("G'", "Ğ"), ("g'", "ğ"), ("SH", "Ş"), ("Sh", "Ş"), ("sh", "ş"),
             ("CH", "Ç"), ("Ch", "Ç"), ("ch", "ç")]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def normalize_apostrophes(text: str) -> str:
    """Unify o'/g' marks and the tutuq to ASCII "'"; drop apostrophes that are really quote marks."""
    keep = "\u0001"  # placeholder, so the clean-up below cannot eat the marks we mean to keep
    text = re.sub(rf"(?<=[oOgG]){_APOS_CLASS}", keep, text)                # o' g' (also word-final: tog', bog')
    text = re.sub(rf"(?<=[A-Za-z]){_APOS_CLASS}(?=[A-Za-z])", keep, text)  # tutuq: ma'no, san'at
    text = re.sub(_APOS_CLASS, "", text)                                   # what is left is a quote mark
    return text.replace(keep, "'")
