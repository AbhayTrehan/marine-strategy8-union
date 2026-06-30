"""
generate.py
===========

Step D of the Strategy 8-U pipeline: Phase II Tri-State Contrastive
Decoding generation (Eq. 17-21). Mirrors the CLI conventions and overall
structure of `marine/generate_llava2.py` as closely as possible (same
`--model_path`/`--image_folder`/`--seed`/`--batch_size` flags, same
`get_chunk` usage for multi-process sharding, same output JSONL schema)
so it slots into this codebase's existing eval tooling (eval/eval_chair.py,
eval/eval_pope.py) without any changes to those files -- but it is its OWN
script, not a modification of generate_llava2.py, which is left untouched.

Reads a "strategy8 question file" produced by build_question_file.py
(Step C) and writes answers in the SAME jsonl schema
`marine/generate_llava2.py` already writes (question_id, image_id, prompt,
text, answer_id, model_id, metadata), so the existing
`eval/eval_chair.py` / `eval/eval_pope.py` work against our output
unmodified.

alpha == 0 fast path: per Eq. 20, alpha=0 gives z_final =
log_softmax(log_softmax(z_ung)), i.e. (after renormalization) exactly the
plain unconditioned generation -- so we skip building the
TriStateGuidanceLogits processor entirely in that case (saving two
redundant forward passes per decoding step), matching
marine/generate_llava2.py's own `if args.guidance_strength == 0` shortcut.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import shortuuid
import torch
from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset import Strategy8TriStateDataset, custom_collate_fn  # noqa: E402
from tristate_logits import TriStateGuidanceLogits  # noqa: E402

from marine.utils.utils import get_chunk  # noqa: E402


def run_generation(model, tokenizer, processor, model_name: str, args) -> None:
    """The actual Phase II generation loop, given an ALREADY-LOADED model.
    Factored out of eval_model() so run_pipeline.py's hyperparameter grid
    search can load the (large) LVLM exactly once and reuse it across every
    trial, instead of paying a full model load per trial."""
    with open(args.question_file) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = args.answers_file
    os.makedirs(os.path.dirname(os.path.abspath(answers_file)), exist_ok=True)
    ans_file = open(answers_file, "w")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Strategy8TriStateDataset(
        questions, args.image_folder, processor, tokenizer, args.conv_mode,
        getattr(model.config, "mm_use_im_start_end", False),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: custom_collate_fn(b, device=device),
    )

    for (
        prompts, question_ids, img_ids,
        ung_ids, pos_ids, neg_ids,
        ung_px, pos_px, neg_px,
        ung_mask, pos_mask, neg_mask,
    ) in loader:

        with torch.inference_mode():
            if args.alpha == 0:
                output_ids = model.generate(
                    ung_ids,
                    pixel_values=ung_px,
                    attention_mask=ung_mask,
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            else:
                output_ids = model.generate(
                    ung_ids,
                    pixel_values=ung_px,
                    attention_mask=ung_mask,
                    do_sample=args.sampling,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([
                        TriStateGuidanceLogits(
                            alpha=args.alpha,
                            guidance_pos_ids=pos_ids,
                            guidance_neg_ids=neg_ids,
                            images_pos=pos_px,
                            images_neg=neg_px,
                            guidance_pos_attention_mask=pos_mask,
                            guidance_neg_attention_mask=neg_mask,
                            model=model,
                            tokenizer=tokenizer,
                        ),
                    ]),
                )

        input_token_len = ung_ids.shape[1]
        decoded_outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        for i, output in enumerate(decoded_outputs):
            output = output.strip()
            print(f"{question_ids[i]}: {output}")

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": question_ids[i],
                "image_id": img_ids[i],
                "prompt": prompts[i],
                "text": output,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {"alpha": args.alpha},
            }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"[Strategy8-U][Step D] Done! Saved answers to {answers_file}")


def load_strategy8_model(model_path: str):
    """Loads the LVLM via the (untouched) original codebase loader, and
    returns (model, tokenizer, processor, model_name) -- a thin wrapper so
    callers (this module's CLI entry point, and run_pipeline.py) share one
    code path for loading."""
    from marine.utils.utils import get_model_name_from_path
    from marine.utils.utils_model import load_model

    model_name = get_model_name_from_path(model_path)
    model, tokenizer, processor = load_model(model_name, model_path)
    return model, tokenizer, processor, model_name


def eval_model(args) -> None:
    """CLI entry point's worker: load the model fresh, then run one
    generation pass. (run_pipeline.py instead calls load_strategy8_model()
    once and run_generation() repeatedly -- see module docstring.)"""
    model, tokenizer, processor, model_name = load_strategy8_model(args.model_path)
    run_generation(model, tokenizer, processor, model_name, args)


def main():
    parser = argparse.ArgumentParser(description="Strategy8-U Step D: tri-state contrastive generation")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    parser.add_argument("--question_file", type=str, required=True,
                        help="strategy8 question file produced by build_question_file.py")
    parser.add_argument("--answers_file", type=str, required=True)

    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--alpha", type=float, default=0.5, help="Eq. 20 guidance strength")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    eval_model(args)


if __name__ == "__main__":
    main()
