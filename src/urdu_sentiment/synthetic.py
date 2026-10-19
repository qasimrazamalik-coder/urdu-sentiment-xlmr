"""SYNTHETIC Urdu and Roman Urdu template sentences.

This fixture exists only so tests and `make smoke` can run without the real dataset.
Sentiment is carried by an obvious adjective, so any model separates it easily. Numbers
measured on this data say nothing about real-world performance and must never be reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_NOTICE = "SYNTHETIC template data: pipeline check only, not a result."

# (subject, grammatical gender). Gender drives adjective and copula agreement.
_URDU_SUBJECTS = [
    ("یہ فلم", "f"),
    ("یہ کھانا", "m"),
    ("اس ہوٹل کی سروس", "f"),
    ("یہ موبائل", "m"),
    ("یہ کتاب", "f"),
    ("ان کا رویہ", "m"),
    ("یہ ڈرامہ", "m"),
    ("یہ دکان", "f"),
]
_URDU_POS = [
    ("اچھا", "اچھی"),
    ("زبردست", "زبردست"),
    ("شاندار", "شاندار"),
    ("بہترین", "بہترین"),
    ("عمدہ", "عمدہ"),
    ("لاجواب", "لاجواب"),
    ("پیارا", "پیاری"),
]
_URDU_NEG = [
    ("برا", "بری"),
    ("بکواس", "بکواس"),
    ("گھٹیا", "گھٹیا"),
    ("ناقص", "ناقص"),
    ("خراب", "خراب"),
    ("فضول", "فضول"),
    ("مایوس کن", "مایوس کن"),
]
_URDU_INTENSIFIERS = ["", "بہت", "واقعی", "انتہائی"]
_URDU_COPULA = {"m": ["ہے", "تھا"], "f": ["ہے", "تھی"]}
_URDU_NOUNS = ["فلم", "کھانے", "ہوٹل", "موبائل", "کتاب", "ڈرامے", "دکان", "سروس"]
_URDU_TIMES = ["آج", "کل", "پرسوں", "اگلے ہفتے"]
_URDU_NEUTRAL = [
    "{t} {n} کے بارے میں میٹنگ ہو گی",
    "{n} کی تفصیلات ویب سائٹ پر موجود ہیں {t}",
    "میں نے {n} کے بارے میں سنا ہے {t}",
    "{t} {n} کے بارے میں بات ہو گی",
]

_ROMAN_SUBJECTS = [
    ("yeh film", "f"),
    ("yeh khana", "m"),
    ("is hotel ki service", "f"),
    ("yeh mobile", "m"),
    ("yeh kitab", "f"),
    ("un ka rawaiya", "m"),
    ("yeh drama", "m"),
    ("yeh dukan", "f"),
]
_ROMAN_POS = [
    ("acha", "achi"),
    ("zabardast", "zabardast"),
    ("shandar", "shandar"),
    ("behtareen", "behtareen"),
    ("umda", "umda"),
    ("lajawab", "lajawab"),
    ("pyara", "pyari"),
]
_ROMAN_NEG = [
    ("bura", "buri"),
    ("bakwas", "bakwas"),
    ("ghatiya", "ghatiya"),
    ("naqis", "naqis"),
    ("kharab", "kharab"),
    ("fuzool", "fuzool"),
    ("mayoos kun", "mayoos kun"),
]
# Several spellings of the same word on purpose: Roman Urdu has no standard orthography.
_ROMAN_INTENSIFIERS = ["", "bohat", "bohot", "boht", "waqai", "intehai"]
_ROMAN_COPULA = {"m": ["hai", "tha"], "f": ["hai", "thi"]}
_ROMAN_NOUNS = ["film", "khane", "hotel", "mobile", "kitab", "drame", "dukan", "service"]
_ROMAN_TIMES = ["aaj", "kal", "parson", "agle hafte"]
_ROMAN_NEUTRAL = [
    "{t} {n} ke baare mein meeting hogi",
    "{n} ki tafseelat website par mojood hain {t}",
    "main ne {n} ke baare mein suna hai {t}",
    "{t} {n} ke baare mein baat hogi",
]


def _opinion(rng: np.random.Generator, urdu: bool, label: str) -> str:
    subjects = _URDU_SUBJECTS if urdu else _ROMAN_SUBJECTS
    pos, neg = (_URDU_POS, _URDU_NEG) if urdu else (_ROMAN_POS, _ROMAN_NEG)
    adjectives = pos if label == "positive" else neg
    intens = _URDU_INTENSIFIERS if urdu else _ROMAN_INTENSIFIERS
    copula = _URDU_COPULA if urdu else _ROMAN_COPULA
    subject, gender = subjects[rng.integers(len(subjects))]
    masc, fem = adjectives[rng.integers(len(adjectives))]
    adj = masc if gender == "m" else fem
    words = [subject, intens[rng.integers(len(intens))], adj, rng.choice(copula[gender])]
    return " ".join(w for w in words if w)


def _neutral(rng: np.random.Generator, urdu: bool) -> str:
    nouns, times, frames = (
        (_URDU_NOUNS, _URDU_TIMES, _URDU_NEUTRAL)
        if urdu
        else (_ROMAN_NOUNS, _ROMAN_TIMES, _ROMAN_NEUTRAL)
    )
    frame = frames[rng.integers(len(frames))]
    text = frame.format(t=times[rng.integers(len(times))], n=nouns[rng.integers(len(nouns))])
    return " ".join(text.split())


def make_synthetic(n_rows: int = 600, seed: int = 0, roman_fraction: float = 0.5) -> pd.DataFrame:
    """Return up to `n_rows` unique labelled sentences with columns text, label, script.

    Sentences are unique so a random split cannot leak identical texts across splits. If the
    template space is smaller than `n_rows` the frame is simply shorter.
    """
    rng = np.random.default_rng(seed)
    labels = ["negative", "neutral", "positive"]
    rows: dict[str, tuple[str, str]] = {}
    attempts = 0
    while len(rows) < n_rows and attempts < n_rows * 40:
        attempts += 1
        label = labels[attempts % 3]
        roman = rng.random() < roman_fraction
        text = _neutral(rng, not roman) if label == "neutral" else _opinion(rng, not roman, label)
        rows.setdefault(text, (label, "roman" if roman else "urdu"))
    frame = pd.DataFrame(
        [(text, label, script) for text, (label, script) in rows.items()],
        columns=["text", "label", "script"],
    )
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
