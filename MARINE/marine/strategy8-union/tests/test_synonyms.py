"""
Run with: python3 tests/test_synonyms.py
(plain-assert script, no pytest dependency required)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synonyms import UnionCanonicalizer, build_raw_mentions, basic_clean, singularize, coco_canonical, load_coco_synonym_map


def _by_canonical(cands):
    return {c.canonical: c for c in cands}


def test_basic_clean_and_singularize():
    assert basic_clean("  A Dog ") == "dog"
    assert basic_clean("the Cell-Phone!!") == "cell phone"
    assert singularize("dogs") == "dog"
    assert singularize("dining tables") == "dining table"
    print("test_basic_clean_and_singularize OK")


def test_coco_canonical_direct_and_head_fallback():
    m = load_coco_synonym_map()
    assert coco_canonical("telephone", m) == "cell phone"
    assert coco_canonical("a young dog", m) == "dog"  # head-word fallback
    assert coco_canonical("xyzzy-not-an-object", m) is None
    print("test_coco_canonical_direct_and_head_fallback OK")


def test_coco_synonyms_merge_across_sources():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["telephone"],
        detr_tags=["cell phone"],
        vlm_objects=["a mobile phone"],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    c = cands[0]
    assert c.canonical == "cell phone"
    assert c.sources == {"ram", "detr", "vlm"}
    assert c.is_coco_category is True
    print("test_coco_synonyms_merge_across_sources OK")


def test_wordnet_merge_for_noncoco_words():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["cloth", "fabric", "material"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    assert cands[0].is_coco_category is False
    assert set(cands[0].raw_mentions) == {"cloth", "fabric", "material"}
    print("test_wordnet_merge_for_noncoco_words OK")


def test_head_noun_merge():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["glass vase", "vase"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1, cands
    # vase is a COCO category, so head-noun-merging a non-coco "glass vase"
    # into it should make the cluster coco-canonical "vase"
    assert cands[0].canonical == "vase"
    assert cands[0].is_coco_category is True
    print("test_head_noun_merge OK")


def test_distinct_coco_categories_never_cross_merge():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["cat", "dog"],
        detr_tags=[],
        vlm_objects=[],
    )
    cands = uc.canonicalize_pool(raws)
    canonicals = {c.canonical for c in cands}
    assert canonicals == {"cat", "dog"}, canonicals
    print("test_distinct_coco_categories_never_cross_merge OK")


def test_empty_pool():
    uc = UnionCanonicalizer()
    assert uc.canonicalize_pool([]) == []
    raws = build_raw_mentions(ram_tags=["", "   ", "123"], detr_tags=[], vlm_objects=[])
    assert uc.canonicalize_pool(raws) == []
    print("test_empty_pool OK")


def test_provenance_tracking():
    uc = UnionCanonicalizer()
    raws = build_raw_mentions(
        ram_tags=["dog"],
        detr_tags=[],
        vlm_objects=["a dog", "a dog"],
    )
    cands = uc.canonicalize_pool(raws)
    assert len(cands) == 1
    assert cands[0].sources == {"ram", "vlm"}
    print("test_provenance_tracking OK")


if __name__ == "__main__":
    test_basic_clean_and_singularize()
    test_coco_canonical_direct_and_head_fallback()
    test_coco_synonyms_merge_across_sources()
    test_wordnet_merge_for_noncoco_words()
    test_head_noun_merge()
    test_distinct_coco_categories_never_cross_merge()
    test_empty_pool()
    test_provenance_tracking()
    print("\nALL synonyms.py TESTS PASSED")
