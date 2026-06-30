"""Run with: python3 tests/test_prompts.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import build_tristate_prompts, objects_to_string


def test_objects_to_string():
    assert objects_to_string([]) == ""
    assert objects_to_string(["dog"]) == "dog"
    assert objects_to_string(["dog", "cat"]) == "dog and cat"
    assert objects_to_string(["dog", "cat", "bird"]) == "dog, cat, and bird"
    print("test_objects_to_string OK")


def test_build_tristate_prompts_normal():
    query = "Generate a short caption of the image."
    c_ung, c_pos, c_neg = build_tristate_prompts(query, ["dog", "person"], ["fork"])
    assert c_ung == query
    assert c_pos == "Focusing on the visible objects in this image: dog and person. generate a short caption of the image."
    assert c_neg == "Focusing on the visible objects in this image: fork. generate a short caption of the image."
    print("test_build_tristate_prompts_normal OK")


def test_empty_object_lists_fall_back_to_query():
    query = "Is there a keyboard in the image?"
    c_ung, c_pos, c_neg = build_tristate_prompts(query, [], [])
    assert c_ung == query
    assert c_pos == query
    assert c_neg == query
    print("test_empty_object_lists_fall_back_to_query OK")


def test_only_negative_nonempty():
    query = "Generate a short caption of the image."
    c_ung, c_pos, c_neg = build_tristate_prompts(query, [], ["fork", "knife"])
    assert c_pos == query  # falls back, no positive evidence
    assert "fork and knife" in c_neg
    print("test_only_negative_nonempty OK")


if __name__ == "__main__":
    test_objects_to_string()
    test_build_tristate_prompts_normal()
    test_empty_object_lists_fall_back_to_query()
    test_only_negative_nonempty()
    print("\nALL prompts.py TESTS PASSED")
