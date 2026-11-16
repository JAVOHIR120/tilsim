"""Which language is a sentence in: Uzbek or English? And cutting mixed text into one-language segments.

    from uznorm.lang import detect_lang, segments
    detect_lang("Describe a place you like to visit.")          # "en"
    segments("Savol shunday: Do you work or are you a student?")
    # [("uz", "Savol shunday:"), ("en", "Do you work or are you a student?")]

Why: English mode (lang="en") reads English with the voice model's own English (no Uzbek accent), Uzbek mode with
the Uzbek LoRA. The switch happens only at a sentence border, or at a ":" when the English part after it is longer
than SHORT_QUOTE words. Shorter English quotes inside an Uzbek sentence («"fluency" va "coherence"») stay in the
Uzbek sentence and are respelled (uznorm/english.py), since a voice switch mid-sentence would be audible.
"""
from __future__ import annotations

import re

from .english import AMBIGUOUS, WORDS, looks_english

SHORT_QUOTE = 3   # English words inside an Uzbek sentence up to this many stay on the Uzbek path

_WORD = re.compile(r"[A-Za-z]+(?:['’ʻ‘][A-Za-z]+)?")
_UZ_MARK = re.compile(r"[og]['’ʻ‘`]|q(?!u)|\bx", re.I)   # o', g', q without u, a word starting with x
# Common English words that are not Uzbek words (a, in, on, to, or, it, is are also Uzbek letters/words: left out).
_EN_COMMON = set("""
the an are was were be been being do does did don't doesn't didn't you your yours you're you've you'll you'd i i'm
i'd i'll i've me my mine we we're we've our ours they they're their them he she his her him its this that these those
what which who whom whose why how when where there here and but if because than then of for with from by about into
over under after before between through during can could would should will shall might must have has had not no yes
very more most some any all each every other like just also please thank thanks let let's now today well good much
many often ever never think say tell give make go going come know want prefer describe explain talk talking discuss
question questions topic minute minutes example little end test speaking listening reading writing really would
something someone people person place time year years day week life work student study home family friend friends
favourite favorite season weather city country remember worry stop right sure okay ok hello goodbye luck
""".split())
# Very common Uzbek words with no Uzbek-only letter (va, bu, endi ...): they mark a sentence as Uzbek.
_UZ_COMMON = set("""
va bu u men sen siz biz ular endi keling yaxshi uchun bilan ham juda emas bor kerak mumkin savol javob iltimos rahmat
qanday nima nega qachon qayerda bir ikki uch shunday lekin ammo yoki agar chunki hozir bugun ertaga kuni haqida yana
mavzu kartasida ajoyib batafsilroq gapirib bering necha soat sizda daqiqa javobingiz keyingi testning qismi boshlanadi
hamma narsa ishlating iborasini takrorladingiz bo'limi imtihonida baholanadi grammatikada vocabulary
""".split()) - {"vocabulary"}


_UZ_END = re.compile(r"(lar|dagi|miz|siz|ingiz|gan|moq|yapti)$")   # Uzbek suffixes (not -ning: morning, evening)
# English word shapes that Uzbek Latin almost never has: final consonant+y (city), final consonant+e (place),
# a word starting with wh, "ou". Used only to score a whole sentence, never to respell a word.
_EN_SHAPE = re.compile(r"[^aeiouy]y$|[^aeiou']e$|^wh|ou")
_EN_ENDS_LIKE_UZ = set("""dollar dollars popular similar regular particular solar polar scholar familiar peculiar
circular muscular spectacular molecular cellular modular pillar collar cellar began organ slogan vegan pagan""".split())


def _tokens(text: str) -> list[str]:
    return [w.lower().replace("’", "'").replace("ʻ", "'").replace("‘", "'") for w in _WORD.findall(text)]


def _score(words: list[str], raw: str) -> tuple[int, int]:
    en = uz = 0
    for w in words:
        if _UZ_MARK.search(w) or w in _UZ_COMMON or (_UZ_END.search(w) and w not in _EN_ENDS_LIKE_UZ):
            uz += 1
        elif w in _EN_COMMON or (w in WORDS and w not in AMBIGUOUS) or looks_english(w) or _EN_SHAPE.search(w):
            en += 1
    return en, uz


def detect_lang(text: str) -> str:
    """'en' when the sentence is English, else 'uz' (the default: Uzbek is what this voice is for)."""
    words = _tokens(text)
    if not words or re.search(r"[Ѐ-ӿ]", text):
        return "uz"
    en, uz = _score(words, text)
    if uz == 0 and en >= 1 and en >= 0.3 * len(words):
        return "en"
    if en >= 3 * max(uz, 1) and en >= 0.6 * len(words):
        return "en"
    return "uz"


def _english_tail(sentence: str) -> tuple[str, str] | None:
    """'Savol shunday: Do you work?' -> ('Savol shunday:', 'Do you work?') when the part after ':' is English."""
    head, sep, tail = sentence.partition(":")
    if not sep or not tail.strip():
        return None
    if detect_lang(tail) == "en" and len(_tokens(tail)) > SHORT_QUOTE and detect_lang(head) == "uz":
        return head.strip() + ":", tail.strip()
    return None


def segments(text: str) -> list[tuple[str, str]]:
    """Mixed text -> [(lang, sentence)] in order: one entry per sentence, or two when a ':' splits off English."""
    out: list[tuple[str, str]] = []
    for sentence in [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]:
        split = _english_tail(sentence)
        parts = [("uz", split[0]), ("en", split[1])] if split else [(detect_lang(sentence), sentence)]
        out += parts
    return out
