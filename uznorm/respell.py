"""Model-only respellings: what the current TTS model needs to SAY Uzbek right, not how Uzbek is written.

Kept apart from normalize() on purpose: normalize() gives the real spoken text (shown in the recording studio, saved as
training transcripts, used by the evaluation), while for_model() is applied only to the text handed to the model.
If the model changes (e.g. after fine-tuning on the owner's voice), re-check every rule here by ear.
"""
from __future__ import annotations

import re

# Word-initial «ya»: the open Uzbek LoRA fronts the «a» towards «e» («yengi», «yexshi»). Written «yaa» the vowel stays open.
# The owner chose «yaa» by ear for yangi, yaxshi (2026-11-14) and for yana, yaqin, yarim, yashil, yagona, yakshanba,
# yashaydi, yaratdi (2026-11-15). Inside a word «ya» is left alone: for «noyabr» he chose the plain spelling.
_INITIAL_YA = re.compile(r"(?<![\w'])([Yy])a(?=[a-z])")  # not «ya'ni»: the tutuq already lengthens that «a»


def for_model(text: str) -> str:
    """Spoken Uzbek text (from normalize()) -> the spelling the TTS model pronounces best."""
    return _INITIAL_YA.sub(lambda m: m.group(1) + "aa", text)
