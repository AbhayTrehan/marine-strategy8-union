"""Run with: python3 tests/test_candidate_pool.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

from candidate_pool import build_pool_record, load_candidate_pool_cache, load_detector_guidance
from synonyms import UnionCanonicalizer


class _FakeFeatureExtractor:
    """Deterministic stand-in for FeatureExtractor.extract: real OWL-ViT/CLIP
    need network access this sandbox doesn't have. Scores are a simple
    deterministic function of the object name so we can assert specific
    candidates end up with specific (recoverable) feature values."""

    def __init__(self):
        self.calls = []

    def extract(self, image, object_names):
        self.calls.append(list(object_names))
        out = {}
        for i, name in enumerate(object_names):
            out[name] = (0.9 - 0.01 * i, 0.3 - 0.005 * i, 0.05)
        return out


def _img():
    return Image.new("RGB", (32, 32), color=(10, 20, 30))


def test_build_pool_record_merges_sources_and_attaches_features():
    canon = UnionCanonicalizer()
    fe = _FakeFeatureExtractor()
    rec = build_pool_record(
        image_file="COCO_val2014_000000000001.jpg",
        pass1_caption="A man is standing near a car and a dog with a cell phone.",
        ram_tags=["telephone", "dog"],
        detr_tags=["car", "person"],
        canonicalizer=canon,
        feature_extractor=fe,
        image=_img(),
    )
    assert rec["image"] == "COCO_val2014_000000000001.jpg"
    by_canon = {c["canonical"]: c for c in rec["candidates"]}

    assert "dog" in by_canon
    assert set(by_canon["dog"]["sources"]) == {"ram", "vlm"}

    assert "cell phone" in by_canon
    assert set(by_canon["cell phone"]["sources"]) == {"ram", "vlm"}
    assert "telephone" in by_canon["cell phone"]["raw_mentions"]

    assert "car" in by_canon
    assert set(by_canon["car"]["sources"]) == {"detr", "vlm"}

    assert "person" in by_canon
    assert set(by_canon["person"]["sources"]) == {"detr", "vlm"}  # "man" -> person

    # every candidate should carry real (non-default-zero) feature scores
    for c in rec["candidates"]:
        assert c["s_det"] > 0
        assert "s_clip" in c and "s_area" in c

    # feature extractor should be called exactly once, with the full
    # canonical object list (batched, not per-object)
    assert len(fe.calls) == 1
    assert set(fe.calls[0]) == set(by_canon.keys())
    print("test_build_pool_record_merges_sources_and_attaches_features OK")


def test_build_pool_record_empty_caption_and_no_detector_tags():
    canon = UnionCanonicalizer()
    fe = _FakeFeatureExtractor()
    rec = build_pool_record(
        image_file="x.jpg",
        pass1_caption="",
        ram_tags=[],
        detr_tags=[],
        canonicalizer=canon,
        feature_extractor=fe,
        image=_img(),
    )
    assert rec["candidates"] == []
    assert fe.calls == []  # feature extractor should not be called for an empty pool
    print("test_build_pool_record_empty_caption_and_no_detector_tags OK")


def test_load_detector_guidance(tmp_path="/tmp/_test_detr_guidance.json"):
    data = [
        {"image": "a.jpg", "objects": ["car", "car", "person"]},
        {"image": "b.jpg", "objects": []},
    ]
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    loaded = load_detector_guidance(tmp_path)
    assert loaded["a.jpg"] == ["car", "car", "person"]
    assert loaded["b.jpg"] == []
    os.remove(tmp_path)
    print("test_load_detector_guidance OK")


def test_candidate_pool_cache_roundtrip():
    canon = UnionCanonicalizer()
    fe = _FakeFeatureExtractor()
    rec1 = build_pool_record("img1.jpg", "A dog and a cat.", ["dog"], ["cat"], canon, fe, image=_img())
    fe2 = _FakeFeatureExtractor()
    rec2 = build_pool_record("img2.jpg", "A red bicycle.", [], ["bicycle"], canon, fe2, image=_img())

    d = tempfile.mkdtemp()
    path = os.path.join(d, "cache.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps(rec1) + "\n")
        f.write(json.dumps(rec2) + "\n")

    loaded = load_candidate_pool_cache(path)
    assert set(loaded.keys()) == {"img1.jpg", "img2.jpg"}
    assert loaded["img1.jpg"]["candidates"] == rec1["candidates"]
    assert loaded["img2.jpg"]["candidates"] == rec2["candidates"]
    print("test_candidate_pool_cache_roundtrip OK")


if __name__ == "__main__":
    test_build_pool_record_merges_sources_and_attaches_features()
    test_build_pool_record_empty_caption_and_no_detector_tags()
    test_load_detector_guidance()
    test_candidate_pool_cache_roundtrip()
    print("\nALL candidate_pool.py TESTS PASSED")
