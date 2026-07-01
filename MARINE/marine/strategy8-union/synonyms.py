"""
synonyms.py
===========

Canonicalization and union-merging for Strategy 8-U's candidate object pool.

The candidate pool O_init = O_det ∪ O_vlm (Strategy8_Union_contrastive.pdf, Eq. 3)
is built from THREE independent, free-text sources that each name objects
slightly differently:

    * RAM++ tags          -- open-vocabulary, e.g. "cell phone", "puppy", "couch"
    * DETR detections      -- fixed 80-class COCO vocabulary, e.g. "cell phone"
    * the VLM's own first-pass caption -- free English text, e.g. "a phone",
      "a dog", "a sofa"

Naively unioning these as raw strings would massively over-count: "puppy",
"dog" and "a dog" would become three different candidate objects instead of
one. The original MARINE codebase already solves a *related* problem (finding
the intersection of DETR/RAM tags for the baseline guidance prompt) with two
tools we reuse here verbatim rather than re-implementing:

    1. `eval/find_intersection.py::synonyms_txt` -- a curated table (taken
       from the CHAIR metric's own synonym list) mapping ~400 common
       words/phrases onto one of the 80 MSCOCO category names. This is the
       SAME table CHAIR uses to score captions, so canonicalizing through it
       keeps our object pool in the exact label space CHAIR will eventually
       evaluate against.
    2. `eval/create_qa.py::get_object_synonyms` -- WordNet noun-synset
       lookup, used there to fuzzily intersect RAM vs. DETR tags beyond the
       curated list.

This module combines both into a single union-merge step: every raw mention,
regardless of source, is normalized and then clustered with any other raw
mention that is either (a) identical after curated-synonym canonicalization,
(b) WordNet-noun-synonymous, or (c) a head-noun match (e.g. "glass vase" and
"vase"). Each cluster becomes one candidate object, tagged with which
source(s) actually mentioned it.

Design note on filtering: we deliberately do NOT hand-filter "non-object"
looking tags (e.g. RAM sometimes emits attributes/actions like "black" or
"sew"). Strategy 8-U's whole premise is that the 3D grounding features
(detector confidence, CLIP similarity, box area) -- not hand-written
heuristics -- are what should decide whether a candidate is real or
hallucinated. A junk tag will simply receive weak grounding evidence and
fall into the negative cluster. Pre-filtering with a brittle noun/POS
heuristic would inject exactly the kind of undocumented heuristic bias the
statistical sorter is supposed to replace. The only filtering done here is
minimal string hygiene (drop empty/pure-punctuation/pure-numeric strings).
"""

from __future__ import annotations

import re
import sys
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from textblob import TextBlob
from nltk.corpus import wordnet as wn

# ---------------------------------------------------------------------------
# Reuse the existing curated synonym table from the original codebase.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "eval"))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from find_intersection import synonyms_txt as _COCO_SYNONYMS_TXT  # noqa: E402
from find_intersection import parse_synonyms as _parse_coco_synonyms  # noqa: E402


# ---------------------------------------------------------------------------
# Physical-object filter for RAM++ / VLM tags
# ---------------------------------------------------------------------------
# Words that pass the WordNet physical-entity check but are noise in RAM++
# output: "photo"/"picture" are meta-references to the image medium itself
# (not objects in the scene); "comfort"/"fill" have an obscure physical noun
# sense in WordNet (quilt; filling material) that RAM++ never uses.
_NON_OBJECT_BLOCKLIST: frozenset = frozenset({
    "photo", "picture", "selfie", "image", "shot",   # meta-image references
    "comfort", "fill",                                 # WordNet edge cases
})

_PHYSICAL_ENTITY_MARKER = "physical_entity.n.01"


@lru_cache(maxsize=4096)
def _has_physical_noun_synset(word: str) -> bool:
    """Returns True if `word` has at least one WordNet noun synset whose
    hypernym closure contains 'physical_entity.n.01'. This distinguishes
    discrete physical objects (bed, cat, blanket) from abstract states
    (sleep, comfort, relax) and pure verbs (sew, lay, take)."""
    key = word.replace(" ", "_")
    for syn in wn.synsets(key, pos=wn.NOUN):
        for hyper in syn.closure(lambda s: s.hypernyms()):
            if hyper.name() == _PHYSICAL_ENTITY_MARKER:
                return True
    return False


def is_likely_physical_object(word: str) -> bool:
    """Combined check used to filter RAM++/VLM tags:
    1. Not in the explicit noise blocklist.
    2. Has at least one WordNet noun synset under physical_entity.n.01.
    COCO-canonical words are always kept before this check is reached
    (since every COCO category is a physical object by definition)."""
    cleaned = word.lower().strip()
    if cleaned in _NON_OBJECT_BLOCKLIST:
        return False
    return _has_physical_noun_synset(cleaned)


_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def load_coco_synonym_map() -> Dict[str, str]:
    """Curated word/phrase -> canonical-of-80-MSCOCO-classes map.

    Identical to what eval/find_intersection.py and eval/eval_chair.py use,
    imported (not duplicated) so the three stay in sync automatically.
    """
    return _parse_coco_synonyms(_COCO_SYNONYMS_TXT)


def basic_clean(text: str) -> str:
    """Lowercase, strip a leading article, drop punctuation, collapse spaces."""
    if text is None:
        return ""
    t = text.strip().lower()
    t = _ARTICLE_RE.sub("", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


@lru_cache(maxsize=4096)
def singularize(word: str) -> str:
    """Rule-based singularization (offline, via TextBlob). Cached: this is
    called a lot during pool construction and TextBlob's rules are pure
    string ops, safe to memoize."""
    if not word:
        return word
    parts = word.split(" ")
    try:
        parts[-1] = TextBlob(parts[-1]).words[0].singularize()
    except IndexError:
        pass
    return " ".join(parts)


def coco_canonical(word: str, synonyms_map: Dict[str, str]) -> Optional[str]:
    """Try to map `word` onto one of the 80 MSCOCO canonical names.

    Tries the full (cleaned, singularized) phrase first, then -- for
    multi-word phrases that aren't a direct hit -- the singularized last
    token, since detectors/VLMs often prefix a head noun with an adjective
    ("young dog", "red bicycle") that isn't itself in the curated table.
    """
    cleaned = basic_clean(word)
    if not cleaned:
        return None
    sing = singularize(cleaned)
    if sing in synonyms_map:
        return synonyms_map[sing]
    if cleaned in synonyms_map:
        return synonyms_map[cleaned]
    tokens = sing.split(" ")
    if len(tokens) > 1:
        head = singularize(tokens[-1])
        if head in synonyms_map:
            return synonyms_map[head]
    return None


@lru_cache(maxsize=4096)
def wordnet_noun_synonyms(word: str) -> frozenset:
    """All WordNet noun-synset lemma names for `word`, plus the word itself.

    Tries the phrase as a single underscore-joined WordNet lookup key first
    (works for compounds WordNet treats as a single lexical entry, e.g.
    "mobile_phone"), then falls back to the last token alone (head-noun
    sense), mirroring the multi-word handling in coco_canonical above.
    """
    if not word:
        return frozenset()
    out: Set[str] = {word}
    tokens = word.split(" ")
    keys = ["_".join(tokens)]
    if len(tokens) > 1:
        keys.append(tokens[-1])
    for key in keys:
        for syn in wn.synsets(key, pos=wn.NOUN):
            for lemma in syn.lemmas():
                out.add(lemma.name().replace("_", " "))
    return frozenset(out)


def _head_word(phrase: str) -> str:
    return phrase.split(" ")[-1] if phrase else phrase


@dataclass
class RawMention:
    text: str          # original, as emitted by the source
    source: str         # 'ram' | 'detr' | 'vlm'
    score: Optional[float] = None  # optional source-native confidence (e.g. DETR softmax prob)


@dataclass
class CandidateObject:
    canonical: str
    sources: Set[str] = field(default_factory=set)
    raw_mentions: List[str] = field(default_factory=list)
    is_coco_category: bool = False

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "sources": sorted(self.sources),
            "raw_mentions": self.raw_mentions,
            "is_coco_category": self.is_coco_category,
        }


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


class UnionCanonicalizer:
    """Merges raw mentions from RAM / DETR / VLM sources into a deduplicated
    list of CandidateObject, using curated-synonym + WordNet union-merging.
    """

    def __init__(self, synonyms_map: Optional[Dict[str, str]] = None):
        self.synonyms_map = synonyms_map or load_coco_synonym_map()

    def _normalize_one(self, raw: str) -> Optional[Tuple[str, bool]]:
        """Returns (label, is_coco) or None if the raw string is empty junk."""
        cleaned = basic_clean(raw)
        if not cleaned:
            return None
        if cleaned.isdigit():
            return None
        coco = coco_canonical(raw, self.synonyms_map)
        if coco is not None:
            return coco, True
        return singularize(cleaned), False

    def canonicalize_pool(self, raw_items: Sequence[RawMention], filter_non_objects: bool = True) -> List[CandidateObject]:
        """Union-merge raw mentions into canonical CandidateObject entries.

        filter_non_objects (default True): drop RAM++ and VLM mentions that
        don't represent discrete physical objects visible in images (e.g.
        "relax", "sleep", "comfort", "photo"). Applied ONLY to RAM and VLM
        sources -- DETR detections are always kept since they come from a
        detection model with a constrained physical-object vocabulary.
        COCO-canonical words (mapped via the curated synonym table) are also
        always kept since every COCO category is a physical object by definition.

        Merge rules (any one is sufficient to join two raw mentions into the
        same cluster):
          (a) identical label after curated-COCO-synonym canonicalization
          (b) WordNet noun-synonym overlap (non-COCO words only)
          (c) head-noun match: a single-word label equals the last token of
              a multi-word label (e.g. "vase" <-> "glass vase")
        """
        entries: List[dict] = []
        for item in raw_items:
            norm = self._normalize_one(item.text)
            if norm is None:
                continue
            label, is_coco = norm

            # Physical-object filter: only for RAM/VLM sources, not DETR,
            # and only for non-COCO words (COCO canonicals are always objects)
            if filter_non_objects and not is_coco and item.source in ("ram", "vlm"):
                if not is_likely_physical_object(label) and not is_likely_physical_object(item.text):
                    continue

            entries.append(
                {"raw": item.text, "source": item.source, "label": label, "is_coco": is_coco}
            )

        n = len(entries)
        if n == 0:
            return []

        uf = UnionFind(n)

        # (a) exact-label grouping (fast path)
        label_groups: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(entries):
            label_groups[e["label"]].append(i)
        for idxs in label_groups.values():
            for i in idxs[1:]:
                uf.union(idxs[0], i)

        # (b) WordNet fuzzy merge, restricted to non-COCO entries only
        noncoco = [i for i, e in enumerate(entries) if not e["is_coco"]]
        syn_cache = {i: wordnet_noun_synonyms(entries[i]["label"]) for i in noncoco}
        for a in range(len(noncoco)):
            for b in range(a + 1, len(noncoco)):
                i, j = noncoco[a], noncoco[b]
                if uf.find(i) == uf.find(j):
                    continue
                if syn_cache[i] & syn_cache[j]:
                    uf.union(i, j)

        # (c) head-noun merge, restricted to non-COCO entries only
        single_word = {i: entries[i]["label"] for i in noncoco if " " not in entries[i]["label"]}
        multi_word = {i: entries[i]["label"] for i in noncoco if " " in entries[i]["label"]}
        for i, lab in single_word.items():
            for j, phrase in multi_word.items():
                if uf.find(i) == uf.find(j):
                    continue
                if _head_word(phrase) == lab:
                    uf.union(i, j)

        # Build clusters
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            clusters[uf.find(i)].append(i)

        candidates: List[CandidateObject] = []
        for idxs in clusters.values():
            members = [entries[i] for i in idxs]
            coco_labels = [m["label"] for m in members if m["is_coco"]]
            if coco_labels:
                canonical = Counter(coco_labels).most_common(1)[0][0]
                is_coco = True
            else:
                # prefer the shortest surface form as the human-readable
                # canonical label (e.g. "vase" over "glass vase")
                labels = [m["label"] for m in members]
                shortest_len = min(len(l) for l in labels)
                shortest_candidates = [l for l in labels if len(l) == shortest_len]
                canonical = Counter(labels).most_common(1)[0][0]
                if canonical not in shortest_candidates:
                    canonical = sorted(shortest_candidates)[0]
                is_coco = False
            sources = {m["source"] for m in members}
            raw_mentions = sorted({m["raw"] for m in members})
            candidates.append(
                CandidateObject(
                    canonical=canonical,
                    sources=sources,
                    raw_mentions=raw_mentions,
                    is_coco_category=is_coco,
                )
            )

        # stable order: COCO categories first (alphabetical), then the rest
        candidates.sort(key=lambda c: (not c.is_coco_category, c.canonical))
        return candidates


def build_raw_mentions(
    ram_tags: Iterable[str],
    detr_tags: Iterable[str],
    vlm_objects: Iterable[str],
) -> List[RawMention]:
    """Convenience constructor for the three sources defined in the paper:
    O_det = f_RAM(I) ∪ f_DETR(I), and O_vlm from the VLM's own first pass.
    """
    out: List[RawMention] = []
    for t in ram_tags:
        out.append(RawMention(text=t, source="ram"))
    for t in detr_tags:
        out.append(RawMention(text=t, source="detr"))
    for t in vlm_objects:
        out.append(RawMention(text=t, source="vlm"))
    return out
