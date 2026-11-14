"""uznorm: turn written Uzbek into the words a TTS voice should say.

    from uznorm import normalize, check, split_sentences
    normalize("2026-yil 14-noyabr")  # -> "ikki ming yigirma oltinchi yil o'n to'rtinchi noyabr"
"""
from .normalize import check, normalize, number_words, reload_tables, split_sentences
from .numbers import cardinal, decimal, fraction, ordinal, roman_to_int, with_suffix
from .respell import for_model
from .translit import cyr_to_lat, has_cyrillic, new_to_old, normalize_apostrophes, old_to_new

__all__ = [
    "normalize", "check", "split_sentences", "number_words", "reload_tables", "for_model",
    "cardinal", "ordinal", "decimal", "fraction", "roman_to_int", "with_suffix",
    "cyr_to_lat", "has_cyrillic", "normalize_apostrophes", "new_to_old", "old_to_new",
]
