"""English inside Uzbek text: find the English stretches and respell them so the Uzbek voice says them the English way.

    from uznorm.english import respell_english
    respell_english("Endi Part ikki ga o'tamiz. Describe a place you like to visit.")
    # -> "Endi Part ikki ga o'tamiz. Diskrayb e pleys yu layk tu vizit."

Why: the voice model was trained on Uzbek, so it reads letters the Uzbek way: «describe» comes out as
«deskribe» and «place» as «plase». Writing English the way an Uzbek reader would spell its sounds gives clear,
Uzbek-accented English, the way a local IELTS teacher speaks. Model-only, like respell.for_model():
the shown and transcribed text stays as written.

What counts as English:
- a word from data/english.tsv, or a word that cannot be Uzbek (w, c without h, th, ph, ee, oo, -tion);
- words marked * in the lexicon (also Uzbek words: a, or, band, test, part, ...) only next to other English words.
Unknown English words inside an English stretch are spelled by rules (respell_word).
"""
from __future__ import annotations

import re
from pathlib import Path

_DATA = Path(__file__).parent / "data" / "english.tsv"
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_UZ_ONLY = re.compile(r"o'|g'|ʻ|‘|’")


def _load() -> tuple[dict[str, str], set[str]]:
    words, ambiguous = {}, set()
    for ln in _DATA.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        en, uz = ln.split("\t")[:2]
        if en.endswith("*"):
            en = en[:-1]
            ambiguous.add(en)
        words[en.lower()] = uz
    return words, ambiguous


WORDS, AMBIGUOUS = _load()


def looks_english(word: str) -> bool:
    """Spelling that Uzbek Latin never uses."""
    w = word.lower()
    # never "ay" (qanday, qulay) or "ea" (okean, real): common in Uzbek
    return bool(re.search(r"w|c(?!h)|th|ph|ee|oo|tion\b|ck|igh", w))


def is_english(word: str) -> bool | None:
    """True / False, or None for words that could be either (decided by their neighbours)."""
    w = word.lower()
    if _UZ_ONLY.search(w):
        return False
    if w in AMBIGUOUS:
        return None
    if w in WORDS or looks_english(w):
        return True
    return False


_RULES = [  # spelling -> sound, applied in order to unknown English words (lower case)
    (r"ation", "eyshn"), (r"tion", "shn"), (r"sion\b", "jn"), (r"ture", "cher"), (r"ph", "f"), (r"th", "t"), (r"ck", "k"),
    (r"qu", "kv"), (r"wh", "v"), (r"w", "v"), (r"x", "ks"), (r"c(?=[eiy])", "s"), (r"c(?!h)", "k"),
    (r"igh", "ay"), (r"ee", "i"), (r"ea", "i"), (r"oo", "u"), (r"ou", "au"), (r"ow\b", "ou"), (r"ai|ay", "ey"),
    (r"au|aw", "o"), (r"ir|ur|er(?=[^aeiou]|$)", "yor"), (r"([aeiou])y\b", r"\1y"), (r"(?<=[^aeiou])y\b", "i"),
    (r"a(?=[^aeiou]e\b)", "ey"), (r"i(?=[^aeiou]e\b)", "ay"), (r"o(?=[^aeiou]e\b)", "ou"), (r"u(?=[^aeiou]e\b)", "yu"),
    (r"(?<=[^aeiou])e\b", ""), (r"j", "dj"), (r"(.)\1", r"\1"),
]


def respell_word(word: str) -> str:
    w = word.lower()
    if w in WORDS:
        out = WORDS[w]
    else:
        out = w
        for pat, rep in _RULES:
            out = re.sub(pat, rep, out)
    return out.capitalize() if word[:1].isupper() else out


def respell_english(text: str) -> str:
    """Respell the English stretches of mixed text; Uzbek words are left exactly as they are."""
    tokens = list(_WORD.finditer(text))
    if not tokens:
        return text
    flags = [is_english(t.group()) for t in tokens]
    # ambiguous words follow their neighbours: English if the word before or after is English
    for i, f in enumerate(flags):
        if f is None:
            prev = next((flags[j] for j in range(i - 1, -1, -1) if flags[j] is not None), False)
            nxt = next((flags[j] for j in range(i + 1, len(flags)) if flags[j] is not None), False)
            flags[i] = bool(prev or nxt) and _same_clause(text, tokens, i)
    out, last = [], 0
    for t, f in zip(tokens, flags):
        out.append(text[last:t.start()])
        out.append(respell_word(t.group()) if f else t.group())
        last = t.end()
    out.append(text[last:])
    return "".join(out)


def _same_clause(text: str, tokens, i: int) -> bool:
    """An ambiguous word takes its neighbour's language only within the same clause (no . ! ? : between)."""
    def joined(a, b):
        return not re.search(r"[.!?:;]", text[a.end():b.start()])
    left = i > 0 and joined(tokens[i - 1], tokens[i])
    right = i + 1 < len(tokens) and joined(tokens[i], tokens[i + 1])
    return left or right
