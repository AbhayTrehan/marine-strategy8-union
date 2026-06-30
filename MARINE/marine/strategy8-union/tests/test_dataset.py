"""
Run with: python3 tests/test_dataset.py

`dataset.py` needs the `llava` package (haotian-liu/LLaVA, cloned manually
per the original README -- it is NOT a pip-installable dependency and is
not available in this sandbox). We inject a tiny mock implementation of
the two bits actually used (`llava.constants`, `llava.conversation`) via
sys.path so `Strategy8TriStateDataset.__getitem__`'s prompt-building logic
can be exercised for real, end to end, against a synthetic processor and a
real (tiny, generated) image file -- everything except the actual neural
network forward passes, which this test suite cannot run in this sandbox.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_mock_llava"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile

import torch
from PIL import Image

from dataset import Strategy8TriStateDataset, custom_collate_fn


class _FakeProcessor:
    """Mimics LlavaProcessor(text=..., images=..., return_tensors='pt')
    closely enough to exercise the Dataset's call pattern: returns
    deterministic-length input_ids (proportional to text length) and a
    fixed-size pixel_values tensor."""

    def __call__(self, text, images, return_tensors="pt"):
        n_tokens = max(3, len(text.split()))
        return {
            "input_ids": torch.arange(n_tokens).unsqueeze(0),
            "attention_mask": torch.ones(1, n_tokens, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 8, 8),
        }


def _make_temp_image_dir():
    d = tempfile.mkdtemp()
    img = Image.new("RGB", (16, 16), color=(120, 50, 200))
    path = os.path.join(d, "COCO_val2014_000000000001.jpg")
    img.save(path)
    return d


def _make_questions():
    return [
        {
            "id": 1,
            "image": "COCO_val2014_000000000001.jpg",
            "conversations": [
                {"from": "human", "value": "Generate a short caption of the image."},
                {"from": "gpt", "value": ""},
                {
                    "from": "guidance_pos",
                    "value": "Focusing on the visible objects in this image: dog and person. generate a short caption of the image.",
                },
                {
                    "from": "guidance_neg",
                    "value": "Focusing on the visible objects in this image: fork. generate a short caption of the image.",
                },
            ],
        },
        {
            "id": 2,
            "image": "COCO_val2014_000000000001.jpg",
            "conversations": [
                {"from": "human", "value": "Is there a keyboard in the image?"},
                {"from": "gpt", "value": ""},
                {"from": "guidance_pos", "value": "Is there a keyboard in the image?"},
                {"from": "guidance_neg", "value": "Is there a keyboard in the image?"},
            ],
        },
    ]


def test_getitem_returns_three_distinct_branches():
    image_dir = _make_temp_image_dir()
    ds = Strategy8TriStateDataset(
        questions=_make_questions(),
        image_dir=image_dir,
        processor=_FakeProcessor(),
        tokenizer=None,
        conv_mode="vicuna_v1",
        mm_use_im_start_end=False,
    )
    item = ds[0]
    (cur_prompt, qid, img_id, ung_ids, pos_ids, neg_ids, ung_px, pos_px, neg_px, ung_mask, pos_mask, neg_mask) = item

    assert qid == 1
    assert img_id == "COCO_val2014_000000000001.jpg"
    assert "<image>" in cur_prompt
    # the pos branch's text is longer (has the object list) than the plain query
    assert pos_ids.shape[1] > ung_ids.shape[1]
    assert neg_ids.shape[1] > ung_ids.shape[1]
    assert ung_px.shape == (1, 3, 8, 8)
    print("test_getitem_returns_three_distinct_branches OK")


def test_getitem_falls_back_when_guidance_equals_query():
    # second question has identical pos/neg/ung text (empty object lists ->
    # prompts.py degrades to the plain query) -- shapes should match.
    image_dir = _make_temp_image_dir()
    ds = Strategy8TriStateDataset(
        questions=_make_questions(),
        image_dir=image_dir,
        processor=_FakeProcessor(),
        tokenizer=None,
        conv_mode="vicuna_v1",
        mm_use_im_start_end=False,
    )
    item = ds[1]
    (_, qid, _, ung_ids, pos_ids, neg_ids, *_rest) = item
    assert qid == 2
    assert ung_ids.shape == pos_ids.shape == neg_ids.shape
    print("test_getitem_falls_back_when_guidance_equals_query OK")


def test_missing_guidance_turn_raises():
    image_dir = _make_temp_image_dir()
    bad_questions = [
        {
            "id": 99,
            "image": "COCO_val2014_000000000001.jpg",
            "conversations": [
                {"from": "human", "value": "Generate a short caption of the image."},
                {"from": "gpt", "value": ""},
                # missing guidance_pos / guidance_neg turns entirely
            ],
        }
    ]
    ds = Strategy8TriStateDataset(
        questions=bad_questions,
        image_dir=image_dir,
        processor=_FakeProcessor(),
        tokenizer=None,
        conv_mode="vicuna_v1",
        mm_use_im_start_end=False,
    )
    try:
        ds[0]
        raise AssertionError("should have raised KeyError")
    except KeyError:
        pass
    print("test_missing_guidance_turn_raises OK")


def test_collate_then_dataset_end_to_end_batch():
    image_dir = _make_temp_image_dir()
    ds = Strategy8TriStateDataset(
        questions=_make_questions(),
        image_dir=image_dir,
        processor=_FakeProcessor(),
        tokenizer=None,
        conv_mode="vicuna_v1",
        mm_use_im_start_end=False,
    )
    batch = [ds[0], ds[1]]
    collated = custom_collate_fn(batch, device="cpu")
    (prompts, qids, img_ids, ung_ids, pos_ids, neg_ids, ung_px, pos_px, neg_px, ung_mask, pos_mask, neg_mask) = collated
    assert qids == [1, 2]
    assert ung_ids.shape[0] == 2
    assert pos_px.shape == (2, 3, 8, 8)
    print("test_collate_then_dataset_end_to_end_batch OK")


if __name__ == "__main__":
    test_getitem_returns_three_distinct_branches()
    test_getitem_falls_back_when_guidance_equals_query()
    test_missing_guidance_turn_raises()
    test_collate_then_dataset_end_to_end_batch()
    print("\nALL dataset.py TESTS PASSED")
