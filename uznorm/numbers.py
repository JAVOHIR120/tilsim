"""Uzbek number words: cardinals, ordinals, decimals, fractions, Roman numerals.

All words use the ASCII apostrophe (o', g'); normalize() swaps it at the end if another form is wanted.
"""
from __future__ import annotations

ONES = ["nol", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
TENS = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]
SCALES = [(10**12, "trillion"), (10**9, "milliard"), (10**6, "million"), (10**3, "ming")]
VOWELS = "aeiou"


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += ["yuz"] if hundreds == 1 else [ONES[hundreds], "yuz"]
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return words


def cardinal(n: int) -> str:
    """2026 -> 'ikki ming yigirma olti'."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n == 0:
        return "nol"
    words: list[str] = []
    for value, name in SCALES:
        count, n = divmod(n, value)
        if not count:
            continue
        if not (count == 1 and value == 1000):  # 1000 is just «ming»
            words.append(cardinal(count))
        words.append(name)
    if n:
        words += _below_thousand(n)
    return " ".join(words)


def _ordinal_word(word: str) -> str:
    if word == "ellik":
        return "elliginchi"  # k -> g before a vowel
    if word[-1] in VOWELS:
        return word + "nchi"
    return word + "inchi"


def ordinal(n: int) -> str:
    """2026 -> 'ikki ming yigirma oltinchi'."""
    words = cardinal(n).split()
    words[-1] = _ordinal_word(words[-1])
    return " ".join(words)


def with_suffix(words: str, suffix: str) -> str:
    """Glue a case suffix onto number words: besh+ga, ellik+ka, qirq+qa, o'n+gacha."""
    if suffix.startswith("g") and words:
        if words.endswith("k"):
            suffix = "k" + suffix[1:]
        elif words.endswith("q"):
            suffix = "q" + suffix[1:]
    return words + suffix


def decimal(int_part: str, frac_part: str) -> str:
    """('2', '5') -> 'ikki butun o'ndan besh'; ('3', '14') -> 'uch butun yuzdan o'n to'rt'."""
    whole = cardinal(int(int_part))
    denominator = cardinal(10 ** len(frac_part))
    return f"{whole} butun {with_suffix(denominator, 'dan')} {cardinal(int(frac_part))}"


def fraction(numerator: int, denominator: int) -> str:
    """3/4 -> 'to'rtdan uch'."""
    return f"{with_suffix(cardinal(denominator), 'dan')} {cardinal(numerator)}"


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def int_to_roman(n: int) -> str:
    out = []
    for value, sym in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                       (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        count, n = divmod(n, value)
        out.append(sym * count)
    return "".join(out)


def roman_to_int(s: str) -> int | None:
    """Strict: returns None unless s is a canonical Roman numeral (so 'IIII' or 'MIM' are rejected)."""
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total = 0
    for i, ch in enumerate(s):
        value = _ROMAN[ch]
        if i + 1 < len(s) and _ROMAN[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total if 0 < total < 4000 and int_to_roman(total) == s else None
