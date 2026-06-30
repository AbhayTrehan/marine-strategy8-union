"""
candidate_pool.py
==================

Step A of the Strategy 8-U pipeline: for every target image, build the
candidate object pool O_init = O_det ∪ O_vlm (Eq. 1-3) and extract the real
3D feature vector x_i = [s_det, s_clip, s_area] for every candidate (Eq. 4-6).

This step is INDEPENDENT of every tunable hyperparameter in
hyperparam_grid.py (the GMM fit, tau, alpha) -- it only depends on the
image itself, the fixed Pass-1 captioning prompt, and the precomputed
DETR/RAM++ guidance files already shipped with this codebase. It is
therefore run exactly ONCE for the full set of images needed (typically
all 500 CHAIR/POPE images), cached to a JSONL file, and reused by every
downstream hyperparameter trial (fit_gmm.py, build_question_file.py) --
this is what keeps the hyperparameter grid search in run_pipeline.py
affordable: re-fitting the GMM and re-classifying candidates is pure numpy
over this cache, no further LVLM/OWL-ViT/CLIP calls.

Detector proposals O_det: per the user's explicit choice, this reuses the
EXISTING precomputed DETR(th=0.95)/RAM++(th=0.68) tag files already in
data/marine_qa/guidance/ rather than re-running those models -- these are
real outputs from an earlier run of marine/grounding_models/{detr,ram}_detect.py,
not synthetic data.

Output schema (one JSON object per line):
{
  "image": "COCO_val2014_000000144305.jpg",
  "pass1_caption": "...",
  "raw": {"ram": [...], "detr": [...], "vlm": [...]},
  "candidates": [
    {"canonical": "dog", "sources": ["ram", "vlm"], "raw_mentions": [...],
     "is_coco_category": true, "s_det": 0.81, "s_clip": 0.29, "s_area": 0.12},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from synonyms import UnionCanonicalizer, build_raw_mentions  # noqa: E402
from text_objects import extract_candidate_nouns  # noqa: E402

PASS1_PROMPT = "Generate a short caption of the image."


# ---------------------------------------------------------------------------
# Pass-1 unguided captioning. Deliberately NOT reusing
# marine/utils/utils_dataset.py::COCOEvalDataset here: that class always
# builds and tokenizes a SECOND ("guidance") prompt per item even when it
# will never be used, which would silently double the image-preprocessing
# cost of this already-expensive step for no benefit, since Pass-1
# captioning (Eq. 2) is, by definition, unguided. This is a deliberately
# minimal, single-prompt dataset instead.
# ---------------------------------------------------------------------------
class _SinglePromptDataset(Dataset):
    def __init__(self, image_files: List[str], image_dir: str, processor, conv_mode: str, mm_use_im_start_end: bool):
        self.image_files = image_files
        self.image_dir = image_dir
        self.processor = processor
        self.conv_mode = conv_mode
        self.mm_use_im_start_end = mm_use_im_start_end

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates

        img_file = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_file)).convert("RGB")

        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        prompt = image_token + "\n" + PASS1_PROMPT
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        return img_file, inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]


def _single_prompt_collate(batch, device="cuda"):
    img_files, ids_list, px_list, mask_list = zip(*batch)

    def process_sequence(seq_list):
        seq_list = [seq.squeeze(0).flip(dims=[0]) for seq in seq_list]
        return pad_sequence(seq_list, batch_first=True, padding_value=0).flip(dims=[1])

    ids = process_sequence(ids_list).to(device)
    mask = process_sequence(mask_list).to(device)
    px = torch.stack(px_list).squeeze(1).to(device)
    return list(img_files), ids, px, mask


def run_pass1_captioning(
    model,
    tokenizer,
    processor,
    conv_mode: str,
    mm_use_im_start_end: bool,
    image_dir: str,
    image_files: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.6,
    top_p: float = 0.9,
    sampling: bool = True,
    device: str = "cuda",
) -> Dict[str, str]:
    """Runs the unguided first pass y^(1) = M_theta(.|c_ung, I) (Eq. 2) for
    every image in `image_files`, batched, with no guidance/contrastive
    machinery at all."""
    dataset = _SinglePromptDataset(image_files, image_dir, processor, conv_mode, mm_use_im_start_end)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: _single_prompt_collate(b, device=device),
    )

    captions: Dict[str, str] = {}
    for img_files, input_ids, pixel_values, attention_mask in loader:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                do_sample=sampling,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        input_len = input_ids.shape[1]
        decoded = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        for img_file, text in zip(img_files, decoded):
            captions[img_file] = text.strip()
    return captions


# ---------------------------------------------------------------------------
# Detector proposal loading (reuses precomputed DETR/RAM++ tag files)
# ---------------------------------------------------------------------------
def load_detector_guidance(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        data = json.load(f)
    return {item["image"]: list(item.get("objects", [])) for item in data}


# ---------------------------------------------------------------------------
# Main per-image pool construction
# ---------------------------------------------------------------------------
def build_pool_record(
    image_file: str,
    pass1_caption: str,
    ram_tags: List[str],
    detr_tags: List[str],
    canonicalizer: UnionCanonicalizer,
    feature_extractor,
    image: Optional[Image.Image] = None,
    image_dir: Optional[str] = None,
) -> dict:
    vlm_objects = extract_candidate_nouns(pass1_caption)
    raw_mentions = build_raw_mentions(ram_tags=ram_tags, detr_tags=detr_tags, vlm_objects=vlm_objects)
    candidates = canonicalizer.canonicalize_pool(raw_mentions)  # Eq. 3 (O_init), with union-merge

    if image is None:
        if image_dir is None:
            raise ValueError("either `image` or `image_dir` must be provided")
        image = Image.open(os.path.join(image_dir, image_file)).convert("RGB")

    object_names = [c.canonical for c in candidates]
    feats = feature_extractor.extract(image, object_names) if object_names else {}

    cand_dicts = []
    for c in candidates:
        s_det, s_clip, s_area = feats.get(c.canonical, (0.0, 0.0, 0.0))
        d = c.to_dict()
        d.update({"s_det": s_det, "s_clip": s_clip, "s_area": s_area})
        cand_dicts.append(d)

    return {
        "image": image_file,
        "pass1_caption": pass1_caption,
        "raw": {"ram": ram_tags, "detr": detr_tags, "vlm": vlm_objects},
        "candidates": cand_dicts,
    }


def build_candidate_pool_cache(
    image_files: List[str],
    image_dir: str,
    detr_guidance_path: str,
    ram_guidance_path: str,
    model,
    tokenizer,
    processor,
    conv_mode: str,
    mm_use_im_start_end: bool,
    feature_extractor,
    output_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.6,
    top_p: float = 0.9,
    sampling: bool = True,
    device: str = "cuda",
) -> None:
    detr_map = load_detector_guidance(detr_guidance_path)
    ram_map = load_detector_guidance(ram_guidance_path)

    print(f"[Strategy8-U][Step A] Running Pass-1 unguided captioning on {len(image_files)} images...")
    captions = run_pass1_captioning(
        model, tokenizer, processor, conv_mode, mm_use_im_start_end,
        image_dir, image_files, batch_size=batch_size, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, sampling=sampling, device=device,
    )

    canonicalizer = UnionCanonicalizer()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_total_candidates = 0
    with open(output_path, "w") as out_f:
        for i, img_file in enumerate(image_files):
            caption = captions.get(img_file, "")
            ram_tags = ram_map.get(img_file, [])
            detr_tags = detr_map.get(img_file, [])

            record = build_pool_record(
                image_file=img_file,
                pass1_caption=caption,
                ram_tags=ram_tags,
                detr_tags=detr_tags,
                canonicalizer=canonicalizer,
                feature_extractor=feature_extractor,
                image_dir=image_dir,
            )
            n_total_candidates += len(record["candidates"])
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(image_files):
                print(f"[Strategy8-U][Step A] {i + 1}/{len(image_files)} images processed "
                      f"({n_total_candidates} candidates so far)")

    print(f"[Strategy8-U][Step A] Done. Cache written to {output_path}")


def load_candidate_pool_cache(path: str) -> Dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["image"]] = rec
    return out


def main():
    parser = argparse.ArgumentParser(description="Strategy8-U Step A: candidate pool + feature extraction")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--image_list_file", type=str, required=True,
                        help="JSON file containing a list of image filenames to process")
    parser.add_argument("--detr_guidance_file", type=str, required=True)
    parser.add_argument("--ram_guidance_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--owlvit_model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sampling", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    from marine.utils.utils import get_model_name_from_path
    from marine.utils.utils_model import load_model
    from feature_extractors import FeatureExtractor

    with open(args.image_list_file) as f:
        image_files = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    model, tokenizer, processor = load_model(model_name, args.model_path)

    feature_extractor = FeatureExtractor(args.owlvit_model, args.clip_model, device=args.device)

    build_candidate_pool_cache(
        image_files=image_files,
        image_dir=args.image_folder,
        detr_guidance_path=args.detr_guidance_file,
        ram_guidance_path=args.ram_guidance_file,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        conv_mode=args.conv_mode,
        mm_use_im_start_end=getattr(model.config, "mm_use_im_start_end", False),
        feature_extractor=feature_extractor,
        output_path=args.output_file,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sampling=args.sampling,
        device=args.device,
    )


if __name__ == "__main__":
    main()
