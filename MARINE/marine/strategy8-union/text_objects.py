"""
text_objects.py
================

Extracts candidate object mentions O_vlm from the LVLM's unguided first-pass
caption y^(1) (Strategy8_Union_contrastive.pdf, Section 2.2).

Key change from the original design: we now use full-sentence POS tagging
(via nltk.pos_tag on the raw, un-singularized tokens) to keep ONLY tokens
tagged as nouns (NN / NNS / NNP / NNPS) in context. This cleanly filters
participial adjectives and gerunds like "sitting" (VBG), "standing" (VBG),
"relax" (VB), etc. that appear in captions but are not physical objects.
The POS tag is determined in the full sentence context (so "sitting" in
"A cat is sitting on a bed" correctly gets VBG, not NN), then double-word
compound nouns (e.g. "teddy bear", "traffic light") are merged and assigned
NN regardless of the individual token tags, since those items only exist in
the noun compound dict.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import nltk
from textblob import TextBlob

try:
    from nltk.corpus import stopwords as _nltk_stopwords
    _STOPWORDS = set(_nltk_stopwords.words("english"))
except LookupError:  # pragma: no cover
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _nltk_stopwords
    _STOPWORDS = set(_nltk_stopwords.words("english"))

_EXTRA_STOPWORDS = {
    "near", "beside", "behind", "atop", "alongside", "amid", "among", "via",
    "plus", "across", "around", "toward", "towards", "upon", "within",
    "throughout", "underneath", "beneath",
    "be", "being", "been", "seem", "seems", "appear", "appears", "look",
    "looks", "shown", "shows", "showing", "feature", "features", "featuring",
    "several", "various", "multiple", "many", "few", "couple",
    "next", "front", "back", "middle", "center", "side",
}
_STOPWORDS = _STOPWORDS | _EXTRA_STOPWORDS

_NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}


def _build_double_word_dict() -> Dict[str, str]:
    coco_double_words = [
        "motor bike", "motor cycle", "air plane", "traffic light", "street light",
        "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
        "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
        "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear",
        "hair drier", "potted plant", "bow tie", "laptop computer", "stove top oven",
        "hot dog", "teddy bear", "home plate", "train track",
    ]
    animal_words = [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "animal", "cub",
    ]
    vehicle_words = ["jet", "train"]

    double_word_dict: Dict[str, str] = {}
    for dw in coco_double_words:
        double_word_dict[dw] = dw
    for aw in animal_words:
        double_word_dict["baby %s" % aw] = aw
        double_word_dict["adult %s" % aw] = aw
    for vw in vehicle_words:
        double_word_dict["passenger %s" % vw] = vw
    double_word_dict["bow tie"] = "tie"
    double_word_dict["toilet seat"] = "toilet"
    double_word_dict["wine glass"] = "wine glass"
    return double_word_dict


_DOUBLE_WORD_DICT = _build_double_word_dict()


def _singularize_token(token: str) -> str:
    try:
        words = TextBlob(token).words
        if len(words) == 1:
            return words[0].singularize()
    except Exception:
        pass
    return token


def _is_candidate_token(word: str) -> bool:
    """Secondary filter: drop stopwords and pure-punctuation/numeric tokens."""
    if not word:
        return False
    if word in _STOPWORDS:
        return False
    if not any(ch.isalpha() for ch in word):
        return False
    if len(word) == 1:
        return False
    return True


def extract_candidate_nouns(caption: str) -> List[str]:
    """Extract noun mentions from a VLM caption using sentence-context POS
    tagging, then double-word merging, then stopword filtering.

    Pipeline:
      1. Tokenize (raw, not singularized) and POS-tag in full sentence
         context for accurate disambiguation (e.g. "sitting" → VBG).
      2. Singularize tokens in parallel, preserving their POS tags.
      3. Merge known double-word compounds (e.g. "teddy"+"bear" →
         "teddy bear", tagged as NN regardless of individual token tags).
      4. MSCOCO special case: drop "seat" after "toilet".
      5. Keep only tokens tagged as NN/NNS/NNP/NNPS that pass the
         secondary stopword filter.
    """
    if not caption or not caption.strip():
        return []

    # Step 1: tokenize + POS tag on the original (un-singularized) tokens
    # for best sentence-context accuracy
    raw_tokens = nltk.word_tokenize(caption.lower())
    if not raw_tokens:
        return []
    tagged_raw: List[Tuple[str, str]] = nltk.pos_tag(raw_tokens)

    # Step 2: singularize in parallel with POS tags
    sing_with_pos: List[Tuple[str, str]] = [
        (_singularize_token(w), pos) for w, pos in tagged_raw
    ]
    sing_words = [w for w, _ in sing_with_pos]

    # Step 3: merge double-word compounds; assign NN to merged items since
    # every entry in _DOUBLE_WORD_DICT is a noun compound by construction
    merged: List[Tuple[str, str]] = []
    i = 0
    while i < len(sing_words):
        if i + 1 < len(sing_words):
            pair = f"{sing_words[i]} {sing_words[i + 1]}"
            if pair in _DOUBLE_WORD_DICT:
                merged.append((_DOUBLE_WORD_DICT[pair], "NN"))
                i += 2
                continue
        merged.append(sing_with_pos[i])
        i += 1

    # Step 4: MSCOCO special case
    words_list = [w for w, _ in merged]
    if "toilet" in words_list and "seat" in words_list:
        merged = [(w, p) for w, p in merged if w != "seat"]

    # Step 5: keep noun-tagged tokens that pass stopword filter
    candidates = [
        word for word, pos in merged
        if pos in _NOUN_TAGS and _is_candidate_token(word)
    ]
    return candidates
