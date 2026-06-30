"""
dataset.py
==========

Tri-state counterpart of `marine/utils/utils_dataset.py::COCOEvalDataset`.
Reads a "strategy8 question file" (built by build_question_file.py) whose
schema extends the original codebase's question-file format with TWO
guidance turns instead of one:

    {
      "id": ...,
      "image": "COCO_val2014_....jpg",
      "conversations": [
        {"from": "human",       "value": "<query, used as c_ung>"},
        {"from": "gpt",         "value": ""},
        {"from": "guidance_pos","value": "<c_pos text, from prompts.py>"},
        {"from": "guidance_neg","value": "<c_neg text, from prompts.py>"}
      ]
    }

Design choice -- processing the image three times: for each of the three
prompts (c_ung, c_pos, c_neg) we call the FULL multimodal processor
(`processor(text=..., images=image, ...)`) independently, exactly as the
original `COCOEvalDataset` does for its two prompts, rather than trying to
reuse a single pixel_values tensor across branches. This costs some
redundant image-preprocessing compute, but LLaVA-style processors expand
the single `<image>` placeholder token into many patch-aligned tokens
*based on what's passed to them jointly* with the image, so tokenizing
text separately from the image risks producing an input_ids tensor that
the model doesn't expect. Since we cannot run the real processor in this
environment to verify a safe shortcut, we mirror the original codebase's
proven-working pattern exactly instead of optimizing this away.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


def _get_turn(conversations: List[dict], from_value: str) -> str:
    for turn in conversations:
        if turn.get("from") == from_value:
            return turn.get("value", "")
    raise KeyError(f"conversation turn '{from_value}' not found")


class Strategy8TriStateDataset(Dataset):
    def __init__(
        self,
        questions: List[dict],
        image_dir: str,
        processor,
        tokenizer,
        conv_mode: str,
        mm_use_im_start_end: bool,
    ):
        self.questions = questions
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end

    def __len__(self) -> int:
        return len(self.questions)

    def _build_full_prompt(self, image_token: str, text: str) -> str:
        from llava.conversation import conv_templates  # lazy: only needed once an LVLM is actually loaded

        prompt = image_token + "\n" + text
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __getitem__(self, idx: int):
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        data = self.questions[idx]
        question_id = data["id"]
        img_id = data.get("image")
        if img_id is None:
            raise ValueError(f"Missing image in question {question_id}")

        image_path = os.path.join(self.image_dir, img_id)
        image = Image.open(image_path).convert("RGB")

        qs = _get_turn(data["conversations"], "human").replace("<image>", "").strip()
        qs_pos = _get_turn(data["conversations"], "guidance_pos")
        qs_neg = _get_turn(data["conversations"], "guidance_neg")

        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        full_prompt_ung = self._build_full_prompt(image_token, qs)
        full_prompt_pos = self._build_full_prompt(image_token, qs_pos)
        full_prompt_neg = self._build_full_prompt(image_token, qs_neg)

        cur_prompt = "<image>\n" + qs

        inputs_ung = self.processor(text=full_prompt_ung, images=image, return_tensors="pt")
        inputs_pos = self.processor(text=full_prompt_pos, images=image, return_tensors="pt")
        inputs_neg = self.processor(text=full_prompt_neg, images=image, return_tensors="pt")

        return (
            cur_prompt,
            question_id,
            img_id,
            inputs_ung["input_ids"],
            inputs_pos["input_ids"],
            inputs_neg["input_ids"],
            inputs_ung["pixel_values"],
            inputs_pos["pixel_values"],
            inputs_neg["pixel_values"],
            inputs_ung["attention_mask"],
            inputs_pos["attention_mask"],
            inputs_neg["attention_mask"],
        )


def custom_collate_fn(batch: List[Tuple], device: str = "cuda") -> Tuple:
    """Tri-state counterpart of utils_dataset.py::custom_collate_fn. Pads
    each of the three input_ids/attention_mask streams independently (they
    generally have different lengths from each other) and stacks the three
    pixel_values streams. All tensors are moved to `device` (defaults to
    "cuda", matching the original codebase's convention; overridable so
    this function can be unit-tested on CPU)."""
    (
        prompts,
        question_ids,
        img_ids,
        ung_ids_list,
        pos_ids_list,
        neg_ids_list,
        ung_px_list,
        pos_px_list,
        neg_px_list,
        ung_mask_list,
        pos_mask_list,
        neg_mask_list,
    ) = zip(*batch)

    def process_sequence(seq_list):
        seq_list = [seq.squeeze(0).flip(dims=[0]) for seq in seq_list]
        return pad_sequence(seq_list, batch_first=True, padding_value=0).flip(dims=[1])

    ung_ids = process_sequence(ung_ids_list).to(device)
    pos_ids = process_sequence(pos_ids_list).to(device)
    neg_ids = process_sequence(neg_ids_list).to(device)

    ung_px = torch.stack(ung_px_list).squeeze(1).to(device)
    pos_px = torch.stack(pos_px_list).squeeze(1).to(device)
    neg_px = torch.stack(neg_px_list).squeeze(1).to(device)

    ung_mask = process_sequence(ung_mask_list).to(device)
    pos_mask = process_sequence(pos_mask_list).to(device)
    neg_mask = process_sequence(neg_mask_list).to(device)

    return (
        list(prompts),
        list(question_ids),
        list(img_ids),
        ung_ids,
        pos_ids,
        neg_ids,
        ung_px,
        pos_px,
        neg_px,
        ung_mask,
        pos_mask,
        neg_mask,
    )
