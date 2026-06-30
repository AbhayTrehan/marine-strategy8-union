"""
text_objects.py
================

Extracts candidate object mentions O_vlm from the LVLM's unguided first-pass
caption y^(1) (Strategy8_Union_contrastive.pdf, Section 2.2):

    "A set of canonical object mentions O_vlm is extracted from y^(1) by
    re-applying the same tagging machinery used to obtain O_det, so that
    VLM-sourced mentions are normalized into the same label space as
    detector proposals."

The "tagging machinery" already present in this codebase is CHAIR's own
tokenization pipeline (eval/eval_chair.py::CHAIR.caption_to_words):
tokenize -> singularize -> merge known double-words (e.g. "teddy" + "bear"
-> "teddy bear") -> drop a couple of MSCOCO-specific special cases (e.g.
"toilet seat" should not fire "chair" via "seat"). We mirror that exact
pipeline here (the double-word table is copied verbatim from
eval/eval_chair.py's CHAIR.__init__ -- see `_build_double_word_dict` --
rather than imported, because instantiating the real `CHAIR` class triggers
loading the full COCO caption/instance annotation files from disk, which
Phase I (this module) has no need for).

Unlike CHAIR itself, we do NOT restrict the output to the 80 MSCOCO
categories: Strategy 8-U explicitly wants to audit hallucinations the VLM
introduces *independently* of the detectors, which may fall outside that
vocabulary. Since we therefore can't rely on "is this token one of the 80
classes?" as our noun filter, we instead drop closed-class function words
(stopwords + a small supplementary set of prepositions/copulas common in
captions) -- a filter on grammatical category, not on "object-likeness",
so it cannot bias which objects look real vs. hallucinated.
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

# Supplementary closed-class words that NLTK's default stopword list misses
# but that commonly appear in image captions and are never themselves
# physical objects (spatial prepositions, copulas/linking verbs, generic
# quantifiers/determiners).
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


def _build_double_word_dict() -> Dict[str, str]:
    """Verbatim copy of the static table built in
    eval/eval_chair.py::CHAIR.__init__ (kept here as plain data so we do not
    need to instantiate the full CHAIR evaluator, which loads COCO
    annotation files from disk that Phase I has no need for)."""
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


def _tokenize_and_singularize(caption: str) -> List[str]:
    tokens = nltk.word_tokenize(caption.lower())
    return [_singularize_token(t) for t in tokens]


def _merge_double_words(words: List[str]) -> List[str]:
    """Identical merge logic to CHAIR.caption_to_words: scan consecutive
    token pairs, replace any pair found in the double-word table with its
    canonical merged form."""
    merged: List[str] = []
    i = 0
    while i < len(words):
        pair = " ".join(words[i:i + 2])
        if pair in _DOUBLE_WORD_DICT:
            merged.append(_DOUBLE_WORD_DICT[pair])
            i += 2
        else:
            merged.append(words[i])
            i += 1
    return merged


def _is_candidate_token(word: str) -> bool:
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
    """Extract a list of candidate object-mention strings from a free-text
    VLM caption, suitable for feeding into
    `synonyms.UnionCanonicalizer.canonicalize_pool` as 'vlm'-sourced
    RawMentions.

    Pipeline (mirrors CHAIR's tagging machinery, see module docstring):
      1. tokenize + singularize (TextBlob, same as CHAIR)
      2. merge known double-words (e.g. "teddy"+"bear" -> "teddy bear"),
         using the exact table CHAIR itself uses
      3. MSCOCO special case: drop a lone "seat" following "toilet" so it
         doesn't separately fire the "chair" synonym group
      4. drop closed-class function words (stopwords)
    """
    if not caption or not caption.strip():
        return []

    words = _tokenize_and_singularize(caption)
    words = _merge_double_words(words)

    if "toilet" in words and "seat" in words:
        words = [w for w in words if w != "seat"]

    candidates = [w for w in words if _is_candidate_token(w)]
    return candidates
