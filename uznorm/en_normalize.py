"""English text -> the words an English voice should say (American reading). Standard library only.

    from uznorm.en_normalize import normalize_en
    normalize_en("The test starts at 9:30 and costs $45, 20% less than in 1998.")
    # -> "The test starts at nine thirty and costs forty-five dollars, twenty percent less than in nineteen ninety-eight."

Used for sentences spoken in English mode (lang="en"). uznorm.normalize() would read digits as Uzbek words.
"""
from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                  "nine": "ninth", "twelve": "twelfth"}
_ABBREV = [(r"\bMr\.", "Mister"), (r"\bMrs\.", "Missus"), (r"\bMs\.", "Miz"), (r"\bDr\.", "Doctor"),
           (r"\bProf\.", "Professor"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
           (r"\betc\.", "et cetera"), (r"\bvs\.?", "versus"), (r"\s&\s", " and ")]
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}


def cardinal_en(n: int) -> str:
    if n < 0:
        return "minus " + cardinal_en(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (" " + cardinal_en(rest) if rest else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            return cardinal_en(head) + " " + name + (" " + cardinal_en(rest) if rest else "")
    raise AssertionError(n)


def ordinal_en(n: int) -> str:
    words = cardinal_en(n)
    head, sep, last = words.rpartition(" " if "-" not in words.rsplit(" ", 1)[-1] else "-")
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_en(n: int) -> str:
    """1998 -> nineteen ninety-eight, 1900 -> nineteen hundred, 2003 -> two thousand three, 2026 -> twenty twenty-six."""
    if 2000 <= n <= 2009:
        return cardinal_en(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal_en(hi) + " hundred"
    return cardinal_en(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else cardinal_en(lo))


def _decimal(int_part: str, frac: str) -> str:
    return cardinal_en(int(int_part)) + " point " + " ".join(_ONES[int(d)] for d in frac)


def _money(m: re.Match) -> str:
    one, many, sub_one, sub_many = _CURRENCY[m.group(1)]
    whole = int(m.group(2).replace(",", ""))
    cents = m.group(3)
    words = f"{cardinal_en(whole)} {one if whole == 1 else many}"
    if cents and int(cents):
        c = int(cents.ljust(2, "0")[:2])
        words += f" and {cardinal_en(c)} {sub_one if c == 1 else sub_many}"
    scale = m.group(4)
    if scale:  # $5 million -> five million dollars
        words = f"{cardinal_en(whole)} {scale.strip().lower()} {many}"
    return words


def _time(m: re.Match) -> str:
    h, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    words = cardinal_en(h) + (" o'clock" if mm == 0 and not ampm else
                              "" if mm == 0 else " oh " + _ONES[mm] if mm < 10 else " " + cardinal_en(mm))
    if ampm:
        words += " " + " ".join(ampm.lower().replace(".", ""))
    return words


def _number(m: re.Match) -> str:
    return _num_words(m.group(0))


def _num_words(raw: str) -> str:
    if "." in raw:
        a, b = raw.replace(",", "").split(".")
        return _decimal(a, b)
    n = int(raw.replace(",", ""))
    if "," not in raw and len(raw) == 4 and 1100 <= n <= 2099:
        return year_en(n)
    return cardinal_en(n)


def normalize_en(text: str) -> str:
    t = " ".join(text.split())
    for pat, rep in _ABBREV:
        t = re.sub(pat, rep, t)
    t = re.sub(r"([$£€])\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(\s(?:million|billion|thousand)\b)?", _money, t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s?([ap]\.?m\.?))?", _time, t, flags=re.I)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal_en(int(m.group(1))), t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: _num_words(m.group(1)) + " percent", t)
    t = re.sub(r"\b(\d+)\s?[-–]\s?(\d+)\b", lambda m: f"{cardinal_en(int(m.group(1)))} to {cardinal_en(int(m.group(2)))}", t)
    t = re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", _number, t)
    return t
