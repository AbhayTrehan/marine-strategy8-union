"""Run with: python3 tests/test_pope_labels.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pope_labels import build_pope_label_file, load_pope_questions


def _write_pope_jsonl(path):
    items = [
        {"question_id": 1, "image": "a.jpg", "text": "Is there a dog?", "label": "yes"},
        {"question_id": 2, "image": "a.jpg", "text": "Is there a cat?", "label": "no"},
        {"question_id": 3, "image": "b.jpg", "text": "Is there a car?", "label": "yes"},
    ]
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def test_load_pope_questions_jsonl():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "pope.json")
    _write_pope_jsonl(path)
    qs = load_pope_questions(path)
    assert len(qs) == 3
    print("test_load_pope_questions_jsonl OK")


def test_build_pope_label_file_no_filter():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "pope.json")
    _write_pope_jsonl(qpath)
    lpath = os.path.join(d, "labels.json")
    n = build_pope_label_file(qpath, lpath)
    assert n == 3
    with open(lpath) as f:
        labels = json.load(f)
    assert labels[0] == {"id": 1, "image": "a.jpg", "label": "yes"}
    print("test_build_pope_label_file_no_filter OK")


def test_build_pope_label_file_with_filter():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "pope.json")
    _write_pope_jsonl(qpath)
    lpath = os.path.join(d, "labels.json")
    n = build_pope_label_file(qpath, lpath, image_filter=["a.jpg"])
    assert n == 2
    with open(lpath) as f:
        labels = json.load(f)
    assert all(l["image"] == "a.jpg" for l in labels)
    print("test_build_pope_label_file_with_filter OK")


if __name__ == "__main__":
    test_load_pope_questions_jsonl()
    test_build_pope_label_file_no_filter()
    test_build_pope_label_file_with_filter()
    print("\nALL pope_labels.py TESTS PASSED")
