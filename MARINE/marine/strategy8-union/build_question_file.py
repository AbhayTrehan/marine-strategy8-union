"""
build_question_file.py
=======================

Step C of the Strategy 8-U pipeline: combines
  (1) candidate_pool.py's cache (O_init + features per image, Step A),
  (2) fit_gmm.py's FROZEN global GMM parameters (Step B), and
  (3) a benchmark's original question file (CHAIR or POPE, unmodified --
      e.g. data/org_qa/chair/coco_chair.json),
into a "strategy8 question file" ready for generate.py (Step D): for every
question, classifies that image's candidates into O_pos/O_neg via Eq. 8
(E-step only, frozen parameters) + the tau threshold (Eq. 15-16), then
builds c_pos/c_neg text (prompts.py) using that SPECIFIC question's actual
query.

Per-image classification is computed ONCE and cached within a single run
of this script (not per-question), so POPE's 6 questions/image only pay
the (already-cheap, pure-numpy) GMM E-step cost once per image.

Output schema (per question):
{
  "id": ...,
  "image": "...",
  "conversations": [
    {"from": "human", "value": "<query>"},
    {"from": "gpt", "value": ""},
    {"from": "guidance_pos", "value": "<c_pos>"},
    {"from": "guidance_neg", "value": "<c_neg>"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np

from gmm import GlobalGMM, GMMParams
from fit_gmm import FeatureScaler
from prompts import build_tristate_prompts


def classify_image_candidates(
    candidates: List[dict],
    gmm: GlobalGMM,
    scaler: FeatureScaler,
    tau: float,
) -> Tuple[List[str], List[str], List[float]]:
    """Applies Eq. 8 (E-step, frozen params) + Eq. 15-16 (tau threshold) to
    one image's candidate list. Returns (o_pos, o_neg, responsibilities).
    The scaler (fitted on the tuning pool) applies sqrt(area) + z-score
    before the GMM E-step -- use_area is read from scaler.use_area."""
    if not candidates:
        return [], [], []
    if scaler.use_area:
        X_raw = np.array([[c["s_det"], c["s_clip"], c["s_area"]] for c in candidates], dtype=float)
    else:
        X_raw = np.array([[c["s_det"], c["s_clip"]] for c in candidates], dtype=float)
    X_norm = scaler.transform(X_raw)
    gamma = gmm.responsibility_positive(X_norm)

    o_pos = [c["canonical"] for c, g in zip(candidates, gamma) if g >= tau]
    o_neg = [c["canonical"] for c, g in zip(candidates, gamma) if g < tau]
    return o_pos, o_neg, [float(g) for g in gamma]


def build_question_file(
    question_path: str,
    candidate_pool_cache: Dict[str, dict],
    gmm: GlobalGMM,
    scaler: FeatureScaler,
    tau: float,
    image_filter: List[str] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Returns (strategy8_questions, per_image_classification).
    scaler encapsulates use_area and the sqrt+z-score transform."""
    try:
        with open(question_path) as f:
            questions = json.load(f)
    except json.JSONDecodeError:
        with open(question_path) as f:
            questions = [json.loads(line) for line in f]

    image_filter_set = set(image_filter) if image_filter is not None else None

    per_image_classification: Dict[str, dict] = {}
    out_questions: List[dict] = []

    for q in questions:
        img = q["image"]
        if image_filter_set is not None and img not in image_filter_set:
            continue

        if img not in per_image_classification:
            rec = candidate_pool_cache.get(img)
            candidates = rec["candidates"] if rec is not None else []
            o_pos, o_neg, gammas = classify_image_candidates(candidates, gmm, scaler, tau)
            per_image_classification[img] = {
                "o_pos": o_pos,
                "o_neg": o_neg,
                "responsibilities": {c["canonical"]: g for c, g in zip(candidates, gammas)},
            }

        cls = per_image_classification[img]

        if "conversations" in q:
            query = q["conversations"][0]["value"]
            qid = q.get("id", q.get("question_id"))
        else:
            query = q["text"]
            qid = q.get("question_id", q.get("id"))

        c_ung, c_pos, c_neg = build_tristate_prompts(query, cls["o_pos"], cls["o_neg"])

        out_questions.append({
            "id": qid,
            "image": img,
            "conversations": [
                {"from": "human", "value": query},
                {"from": "gpt", "value": ""},
                {"from": "guidance_pos", "value": c_pos},
                {"from": "guidance_neg", "value": c_neg},
            ],
        })

    return out_questions, per_image_classification


def main():
    from candidate_pool import load_candidate_pool_cache

    parser = argparse.ArgumentParser(description="Strategy8-U Step C: build the tri-state question file")
    parser.add_argument("--question_file", type=str, required=True,
                        help="original benchmark question file, e.g. data/org_qa/chair/coco_chair.json")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--gmm_params_file", type=str, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--image_list_file", type=str, default=None,
                        help="optional JSON list of image filenames to restrict to (e.g. tune/test split)")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--classification_output_file", type=str, default=None,
                        help="optional: dump per-image O_pos/O_neg/responsibilities (useful for the HTML report)")
    args = parser.parse_args()

    cache = load_candidate_pool_cache(args.candidate_pool_cache)
    gmm = GlobalGMM.from_params(GMMParams.load(args.gmm_params_file))

    image_filter = None
    if args.image_list_file:
        with open(args.image_list_file) as f:
            image_filter = json.load(f)

    out_questions, per_image = build_question_file(
        args.question_file, cache, gmm, args.tau, image_filter=image_filter,
    )

    with open(args.output_file, "w") as f:
        json.dump(out_questions, f, indent=2)
    print(f"[Strategy8-U][Step C] Wrote {len(out_questions)} questions ({len(per_image)} images) "
          f"to {args.output_file} (tau={args.tau})")

    if args.classification_output_file:
        with open(args.classification_output_file, "w") as f:
            json.dump(per_image, f, indent=2)


if __name__ == "__main__":
    main()
