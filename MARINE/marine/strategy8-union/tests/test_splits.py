"""Run with: python3 tests/test_splits.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splits import ImageSplit, make_split


def _fake_images(n=500):
    return [f"COCO_val2014_{str(i).zfill(12)}.jpg" for i in range(1, n + 1)]


def test_split_sizes():
    images = _fake_images(500)
    split = make_split(images, n_tune=300, n_report=100, seed=8)
    assert len(split.tune_images) == 300
    assert len(split.test_images) == 200
    assert len(split.report_images) == 100
    print("test_split_sizes OK")


def test_split_disjoint_and_covers_all():
    images = _fake_images(500)
    split = make_split(images, n_tune=300, n_report=100, seed=8)
    tune_set = set(split.tune_images)
    test_set = set(split.test_images)
    assert tune_set.isdisjoint(test_set)
    assert tune_set | test_set == set(images)
    assert set(split.report_images) <= test_set
    print("test_split_disjoint_and_covers_all OK")


def test_split_deterministic_given_seed():
    images = _fake_images(500)
    s1 = make_split(images, seed=8)
    s2 = make_split(images, seed=8)
    assert s1.tune_images == s2.tune_images
    assert s1.test_images == s2.test_images
    assert s1.report_images == s2.report_images
    s3 = make_split(images, seed=123)
    assert s1.tune_images != s3.tune_images
    print("test_split_deterministic_given_seed OK")


def test_split_robust_to_input_order_and_dupes():
    images = _fake_images(500)
    shuffled = images[::-1]
    with_dupes = images + images[:10]
    s1 = make_split(images, seed=8)
    s2 = make_split(shuffled, seed=8)
    s3 = make_split(with_dupes, seed=8)
    assert s1.tune_images == s2.tune_images == s3.tune_images
    print("test_split_robust_to_input_order_and_dupes OK")


def test_split_serialization_roundtrip(tmp_path="/tmp/_test_split.json"):
    images = _fake_images(500)
    s1 = make_split(images, seed=8)
    s1.save(tmp_path)
    s2 = ImageSplit.load(tmp_path)
    assert s1.to_dict() == s2.to_dict()
    os.remove(tmp_path)
    print("test_split_serialization_roundtrip OK")


def test_invalid_sizes_raise():
    images = _fake_images(10)
    try:
        make_split(images, n_tune=20)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    try:
        make_split(images, n_tune=5, n_report=10)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_invalid_sizes_raise OK")


if __name__ == "__main__":
    test_split_sizes()
    test_split_disjoint_and_covers_all()
    test_split_deterministic_given_seed()
    test_split_robust_to_input_order_and_dupes()
    test_split_serialization_roundtrip()
    test_invalid_sizes_raise()
    print("\nALL splits.py TESTS PASSED")
