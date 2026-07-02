"""
run_pipeline.py
================

Top-level orchestrator for the Strategy 8-U pipeline (items #1, #5, #7 of
the spec). End-to-end flow:

  Step A (candidate_pool.py, run ONCE for all images, cached to disk):
      build O_init = O_det u O_vlm and extract real x_i = [s_det, s_clip,
      s_area] for every candidate, for every image in the dataset.

  [optional, toggled by --tune] Hyperparameter search (item #5), now in
  TWO decoupled phases:
      B'. GMM PRESET SELECTION (gmm_selection.py): every candidate GMM
          preset (init strategy / means / covariance, + optionally
          learning_rate if --tune_learning_rate) is fit on the pooled
          tuning-image features and scored with an INTRINSIC, label-free
          cluster-quality metric (silhouette score) -- no LVLM generation
          at all. The best one is frozen and used for every trial below.
      C+D. (tau, alpha) GRID, using that ONE fixed GMM: each distinct
          (tau, alpha) combination requires a real generation pass over
          the tuning images, scored by CHAIR -> F1 = 2PR/(P+R),
          P = 1 - CHAIRi, R = CHAIR Recall. Every trial's result is
          persisted to disk under a name derived purely from (tau, alpha)
          (not run-order), so re-running --tune skips any trial whose
          result is already on disk instead of recomputing it.

  Final evaluation (item #7), using the (just-tuned, or previously-tuned-
  and-now-loaded) winning hyperparameters: CHAIR and POPE, each on the
  200 held-out TEST images and again on the FULL 500 (stored separately).

  Report (item #6): built entirely from the CHAIR test-200 run's already-
  produced artifacts -- no model calls needed at all.

--stage controls which of the above actually run in a given invocation
(see build_arg_parser): 'all' (default, everything), 'tune_only' (just
the hyperparameter search), 'chair' (just the CHAIR final eval), 'pope'
(just the POPE final eval), 'report' (just the report -- and uniquely,
this path never loads the LVLM at all, since it only reads already-
generated files).

Nothing in this file modifies any file outside marine/strategy8-union/; it
only ever *calls* the original codebase's untouched modules
(marine.utils.utils_model.load_model, eval.eval_chair.CHAIR,
eval.eval_pope's helpers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_MARINE_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if _MARINE_ROOT not in sys.path:
    sys.path.insert(0, _MARINE_ROOT)
_EVAL_DIR = os.path.join(_MARINE_ROOT, "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from splits import ImageSplit, make_split  # noqa: E402
from hyperparam_grid import (  # noqa: E402
    TrialConfig, TrialResult, build_grid, chair_f1, pick_best, select_gmm_presets,
    BASE_GMM_PRESETS, DAMPED_GMM_PRESETS,
)
from gmm import GlobalGMM, GMMParams  # noqa: E402
from fit_gmm import FeatureScaler  # noqa: E402
from gmm_selection import GMMSelectionResult, select_best_gmm_preset  # noqa: E402
from fit_gmm import fit_global_gmm  # noqa: E402
from build_question_file import build_question_file  # noqa: E402
from candidate_pool import build_candidate_pool_cache, load_candidate_pool_cache  # noqa: E402
from pope_labels import build_pope_label_file  # noqa: E402
import generate  # noqa: E402


STAGE_CHOICES = ["all", "tune_only", "set_hyperparams", "chair", "pope", "report"]


# ---------------------------------------------------------------------------
# Step A: candidate pool cache (run once, reused by every downstream step)
# ---------------------------------------------------------------------------
def ensure_candidate_pool_cache(
    args, model, tokenizer, processor, feature_extractor_factory, image_files: List[str],
) -> Dict[str, dict]:
    cache_path = os.path.join(args.output_dir, "candidate_pool_cache.jsonl")
    if os.path.exists(cache_path) and not args.force_recompute_pool:
        print(f"[Strategy8-U] Reusing existing candidate pool cache: {cache_path}")
        return load_candidate_pool_cache(cache_path)

    print(f"[Strategy8-U] Building candidate pool cache for {len(image_files)} images -> {cache_path}")
    feature_extractor = feature_extractor_factory()  # OWL-ViT/CLIP only loaded when actually needed
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
        output_path=cache_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sampling=args.sampling,
        device=args.device,
    )
    return load_candidate_pool_cache(cache_path)


# ---------------------------------------------------------------------------
# A single (tau, alpha) trial against a FIXED, already-chosen GMM: classify
# + prompt (Step C), generate (Step D), evaluate.
# ---------------------------------------------------------------------------
def run_one_chair_trial(
    trial: TrialConfig,
    cache: Dict[str, dict],
    images: List[str],
    chair_question_file: str,
    model, tokenizer, processor, model_name,
    chair_evaluator,
    work_dir: str,
    gen_args_template,
    prefit_gmm: Optional[GlobalGMM] = None,
    prefit_scaler: Optional[FeatureScaler] = None,
):
    os.makedirs(work_dir, exist_ok=True)

    use_area = prefit_scaler.use_area if prefit_scaler is not None else True
    if prefit_gmm is not None and prefit_scaler is not None:
        gmm, scaler = prefit_gmm, prefit_scaler
    else:
        gmm, scaler = fit_global_gmm(cache, images, trial.gmm_preset, use_area=use_area)
    gmm_params_path = os.path.join(work_dir, "gmm_params.json")
    gmm.params.save(gmm_params_path)

    questions, _ = build_question_file(
        chair_question_file, cache, gmm, scaler, trial.tau, image_filter=images,
    )
    qfile_path = os.path.join(work_dir, "question_file.json")
    with open(qfile_path, "w") as f:
        json.dump(questions, f)

    answers_path = os.path.join(work_dir, "answers.jsonl")
    gen_args = _clone_args(gen_args_template, question_file=qfile_path, answers_file=answers_path, alpha=trial.alpha)
    generate.run_generation(model, tokenizer, processor, model_name, gen_args)

    result = chair_evaluator.compute_chair(answers_path, "image_id", "text")
    metrics = result["overall_metrics"]
    f1 = chair_f1(metrics["CHAIRi"], metrics["Recall"])

    eval_out_path = os.path.join(work_dir, "chair_eval.json")
    with open(eval_out_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "hallucinated_caps_ls"}, f, indent=2)

    trial_result = TrialResult(
        trial=trial,
        chair_s=metrics["CHAIRs"],
        chair_i=metrics["CHAIRi"],
        recall=metrics["Recall"],
        f1=f1,
        n_images=int(metrics["num_caps"]),
        extra={"answers_path": answers_path, "gmm_params_path": gmm_params_path},
    )
    return trial_result, gmm.params, scaler


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _clone_args(template, **overrides):
    d = dict(vars(template))
    d.update(overrides)
    return _Args(**d)


def _gen_args_template(args) -> _Args:
    return _Args(
        image_folder=args.image_folder,
        conv_mode=args.conv_mode,
        num_chunks=1,
        chunk_idx=0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        sampling=args.sampling,
    )


def _sanitize_trial_id(trial_id: str) -> str:
    return trial_id.replace("/", "_")


# ---------------------------------------------------------------------------
# Hyperparameter search (item #5): GMM preset selection (cheap, decoupled,
# cached) followed by a (tau, alpha) grid (expensive, per-trial cached).
# ---------------------------------------------------------------------------
def run_hyperparameter_search(
    args, cache, split: ImageSplit, model, tokenizer, processor, model_name, chair_evaluator,
) -> dict:
    tune_root = os.path.join(args.output_dir, "tuning")
    os.makedirs(tune_root, exist_ok=True)

    # ---- Phase B': GMM preset selection -- intrinsic quality, NO generation ----
    gmm_selection_path = os.path.join(tune_root, "gmm_selection.json")
    candidate_presets = select_gmm_presets(args.tune_learning_rate, use_area=(not args.no_area_feature))
    if os.path.exists(gmm_selection_path) and not args.force_recompute_trials:
        selection = GMMSelectionResult.load(gmm_selection_path)
        print(f"[Strategy8-U][Tune] Reusing cached GMM preset selection: '{selection.chosen_preset_name}' "
              f"(delete {gmm_selection_path} or pass --force_recompute_trials to redo this).")
    else:
        print(f"[Strategy8-U][Tune] Selecting GMM preset among {[p['name'] for p in candidate_presets]} "
              f"via intrinsic fit quality (silhouette score) on {len(split.tune_images)} tuning images "
              f"-- no LVLM generation involved in this step.")
        selection = select_best_gmm_preset(cache, split.tune_images, candidate_presets, use_area=(not args.no_area_feature))
        selection.save(gmm_selection_path)
        for name, q in selection.quality_by_preset.items():
            marker = " <= CHOSEN" if name == selection.chosen_preset_name else ""
            print(f"    {name}: silhouette={q['silhouette']:.4f} separation={q['mean_separation']:.4f} "
                  f"loglik={q['log_likelihood']:.2f}{marker}")

    chosen_preset = selection.chosen_preset
    chosen_gmm = GlobalGMM.from_params(selection.chosen_gmm_params)
    chosen_scaler = selection.chosen_scaler

    # ---- Phase C+D: (tau, alpha) grid against the FIXED chosen GMM ----
    preferred_first = None
    if args.first_tau is not None or args.first_alpha is not None:
        preferred_first = {"tau": args.first_tau, "alpha": args.first_alpha}

    trials = build_grid(
        gmm_presets=[chosen_preset],
        taus=args.taus, alphas=args.alphas,
        max_trials=args.max_trials, seed=args.grid_seed,
        preferred_first=preferred_first,
    )
    print(f"[Strategy8-U][Tune] Evaluating {len(trials)} (tau, alpha) trials on "
          f"{len(split.tune_images)} tuning images (GMM preset fixed to '{chosen_preset['name']}')...")

    gen_args_template = _gen_args_template(args)
    results: List[TrialResult] = []

    for i, trial in enumerate(trials):
        work_dir = os.path.join(tune_root, f"trial__{_sanitize_trial_id(trial.trial_id)}")
        cached_result_path = os.path.join(work_dir, "trial_result.json")

        if os.path.exists(cached_result_path) and not args.force_recompute_trials:
            with open(cached_result_path) as f:
                result = TrialResult.from_dict(json.load(f))
            print(f"[Strategy8-U][Tune] Trial {i + 1}/{len(trials)}: {trial.trial_id} "
                  f"-- REUSING cached result (F1={result.f1:.4f}); pass --force_recompute_trials to redo.")
        else:
            print(f"[Strategy8-U][Tune] Trial {i + 1}/{len(trials)}: {trial.trial_id}")
            result, _, _ = run_one_chair_trial(
                trial, cache, split.tune_images, args.chair_question_file,
                model, tokenizer, processor, model_name, chair_evaluator,
                work_dir, gen_args_template, prefit_gmm=chosen_gmm, prefit_scaler=chosen_scaler,
            )
            with open(cached_result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"[Strategy8-U][Tune]   CHAIRs={result.chair_s:.4f} CHAIRi={result.chair_i:.4f} "
                  f"Recall={result.recall:.4f} F1={result.f1:.4f}")
        results.append(result)

    best = pick_best(results)

    all_trials_path = os.path.join(tune_root, "all_trials.json")
    with open(all_trials_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    best_hyperparams = {
        "trial": best.trial.to_dict(),
        "gmm_params": selection.chosen_gmm_params.to_dict(),
        "feature_scaler": selection.chosen_scaler.to_dict(),
        "tuning_result": best.to_dict(),
        "gmm_selection_quality": selection.quality_by_preset,
    }
    with open(args.best_hyperparams_file, "w") as f:
        json.dump(best_hyperparams, f, indent=2)

    print(f"[Strategy8-U][Tune] Best trial: {best.trial.trial_id} (F1={best.f1:.4f}). "
          f"Saved to {args.best_hyperparams_file}")
    return best_hyperparams


def load_best_hyperparams(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Final evaluation (item #7): held-out 200 + full 500, CHAIR and POPE
# ---------------------------------------------------------------------------
def run_final_chair_eval(
    args, cache, images: List[str], tag: str,
    gmm: GlobalGMM, scaler: FeatureScaler, tau: float, alpha: float,
    model, tokenizer, processor, model_name, chair_evaluator,
) -> dict:
    out_dir = os.path.join(args.output_dir, "final")
    os.makedirs(out_dir, exist_ok=True)

    questions, per_image = build_question_file(
        args.chair_question_file, cache, gmm, scaler, tau, image_filter=images,
    )
    qfile_path = os.path.join(out_dir, f"chair_question_file_{tag}.json")
    with open(qfile_path, "w") as f:
        json.dump(questions, f)
    classification_path = os.path.join(out_dir, f"chair_classification_{tag}.json")
    with open(classification_path, "w") as f:
        json.dump(per_image, f, indent=2)

    answers_path = os.path.join(out_dir, f"chair_answers_{tag}.jsonl")
    gen_args = _clone_args(_gen_args_template(args), question_file=qfile_path, answers_file=answers_path, alpha=alpha)
    generate.run_generation(model, tokenizer, processor, model_name, gen_args)

    result = chair_evaluator.compute_chair(answers_path, "image_id", "text")
    metrics = result["overall_metrics"]
    eval_path = os.path.join(out_dir, f"chair_eval_{tag}.json")
    with open(eval_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "hallucinated_caps_ls"}, f, indent=2)

    print(f"[Strategy8-U][Final][CHAIR/{tag}] CHAIRs={metrics['CHAIRs']:.4f} "
          f"CHAIRi={metrics['CHAIRi']:.4f} Recall={metrics['Recall']:.4f}")

    return {
        "question_file": qfile_path,
        "classification_file": classification_path,
        "answers_file": answers_path,
        "eval_file": eval_path,
        "metrics": metrics,
    }


def run_final_pope_eval(
    args, cache, images: List[str], tag: str,
    gmm: GlobalGMM, scaler: FeatureScaler, tau: float, alpha: float,
    model, tokenizer, processor, model_name,
) -> dict:
    from eval_pope import load_labels, load_predictions, compute_metrics  # original codebase, unmodified

    out_dir = os.path.join(args.output_dir, "final")
    os.makedirs(out_dir, exist_ok=True)

    questions, per_image = build_question_file(
        args.pope_question_file, cache, gmm, scaler, tau, image_filter=images,
    )
    qfile_path = os.path.join(out_dir, f"pope_question_file_{tag}.json")
    with open(qfile_path, "w") as f:
        json.dump(questions, f)

    label_path = os.path.join(out_dir, f"pope_labels_{tag}.json")
    build_pope_label_file(args.pope_question_file, label_path, image_filter=images)

    answers_path = os.path.join(out_dir, f"pope_answers_{tag}.jsonl")
    gen_args = _clone_args(_gen_args_template(args), question_file=qfile_path, answers_file=answers_path, alpha=alpha)
    generate.run_generation(model, tokenizer, processor, model_name, gen_args)

    labels = load_labels(label_path)
    preds, answers = load_predictions(answers_path, model="strategy8_union")
    if len(labels) != len(answers):
        print(f"[Strategy8-U][Final][POPE/{tag}][WARNING] label count ({len(labels)}) "
              f"!= answer count ({len(answers)}); metrics may be misaligned.")
    metrics = compute_metrics(labels, preds, answers, answers_path)

    eval_path = os.path.join(out_dir, f"pope_eval_{tag}.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    om = metrics["overall_metrics"]
    print(f"[Strategy8-U][Final][POPE/{tag}] Accuracy={om['Accuracy']:.4f} "
          f"F1={om['F1']:.4f} Yes_ratio={om['Yes_ratio']:.4f}")

    return {
        "question_file": qfile_path,
        "label_file": label_path,
        "answers_file": answers_path,
        "eval_file": eval_path,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# summary.json: accumulated across separate --stage invocations, never
# wholesale-overwritten, so e.g. a later `--stage pope` run doesn't erase
# what an earlier `--stage chair` run already recorded.
# ---------------------------------------------------------------------------
def update_summary(output_dir: str, **updates) -> dict:
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary.update({k: v for k, v in updates.items() if v is not None})
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# --stage set_hyperparams: directly fix (tau, alpha) to known values, with
# NO generation and NO evaluation -- just fits the chosen GMM preset and
# writes best_hyperparams.json. Use this when you already know exactly
# which (tau, alpha) you want and don't need the grid search at all.
# Lightweight: loads the LVLM only if the Step A cache doesn't exist yet.
# ---------------------------------------------------------------------------
def run_set_hyperparams_stage(args) -> None:
    if args.fixed_tau is None or args.fixed_alpha is None:
        raise ValueError("--stage set_hyperparams requires both --fixed_tau and --fixed_alpha.")

    all_presets = {p["name"]: p for p in (BASE_GMM_PRESETS + DAMPED_GMM_PRESETS)}
    if args.gmm_preset_name not in all_presets:
        raise ValueError(f"--gmm_preset_name must be one of {list(all_presets)}, got {args.gmm_preset_name!r}")
    preset = all_presets[args.gmm_preset_name]

    with open(args.chair_question_file) as f:
        chair_questions = json.load(f)
    all_images = sorted({q["image"] for q in chair_questions})

    split_path = os.path.join(args.output_dir, "split.json")
    if os.path.exists(split_path):
        split = ImageSplit.load(split_path)
    else:
        split = make_split(all_images, n_tune=args.n_tune_images, n_report=args.n_report_images, seed=args.split_seed)
        split.save(split_path)
        print(f"[Strategy8-U] Created new split ({len(split.tune_images)} tune / "
              f"{len(split.test_images)} test / {len(split.report_images)} report) -> {split_path}")

    cache_path = os.path.join(args.output_dir, "candidate_pool_cache.jsonl")
    if os.path.exists(cache_path) and not args.force_recompute_pool:
        cache = load_candidate_pool_cache(cache_path)
        print(f"[Strategy8-U] Reusing existing candidate pool cache: {cache_path} (no LVLM loaded for this stage).")
    else:
        print("[Strategy8-U] Candidate pool cache not found yet -- building it first "
              "(this DOES require loading the LVLM once, but only this one time).")
        model, tokenizer, processor, model_name = generate.load_strategy8_model(args.model_path)

        def _feature_extractor_factory():
            from feature_extractors import FeatureExtractor
            return FeatureExtractor(args.owlvit_model, args.clip_model, device=args.device)

        cache = ensure_candidate_pool_cache(args, model, tokenizer, processor, _feature_extractor_factory, all_images)

    print(f"[Strategy8-U] Fitting GMM preset '{args.gmm_preset_name}' on {len(split.tune_images)} tuning images "
          f"(pure numpy -- no generation, no evaluation).")
    gmm = fit_global_gmm(cache, split.tune_images, preset)

    best_hyperparams = {
        "trial": {
            "trial_id": f"{preset['name']}__tau{args.fixed_tau}__alpha{args.fixed_alpha}__manual",
            "gmm_preset": preset,
            "tau": args.fixed_tau,
            "alpha": args.fixed_alpha,
        },
        "gmm_params": gmm.params.to_dict(),
        "tuning_result": None,
        "note": "Set directly via --stage set_hyperparams; no grid search or tuning-set "
                "evaluation was performed, so there is no F1/CHAIR score to report here.",
    }
    with open(args.best_hyperparams_file, "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"[Strategy8-U] Wrote {args.best_hyperparams_file}: tau={args.fixed_tau}, alpha={args.fixed_alpha}, "
          f"gmm_preset='{args.gmm_preset_name}'. Run --stage chair / pope / report next.")


# ---------------------------------------------------------------------------
# --stage report: lightweight, reads already-generated files only, never
# loads the LVLM/CHAIR-evaluator/feature-extractors at all.
# ---------------------------------------------------------------------------
def run_report_stage(args) -> None:
    out_dir = os.path.join(args.output_dir, "final")
    qfile_path = os.path.join(out_dir, "chair_question_file_test200.json")
    classification_path = os.path.join(out_dir, "chair_classification_test200.json")
    answers_path = os.path.join(out_dir, "chair_answers_test200.jsonl")
    missing = [p for p in [qfile_path, classification_path, answers_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "--stage report needs the CHAIR test200 run's artifacts, which don't exist yet: "
            f"{missing}. Run `--stage chair` (or `--stage all`) first."
        )

    split_path = os.path.join(args.output_dir, "split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"--stage report needs {split_path} (created by any earlier stage).")
    split = ImageSplit.load(split_path)

    cache_path = os.path.join(args.output_dir, "candidate_pool_cache.jsonl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"--stage report needs {cache_path} (created by Step A in any earlier stage).")
    cache = load_candidate_pool_cache(cache_path)

    config_info = {}
    if os.path.exists(args.best_hyperparams_file):
        best_hyperparams = load_best_hyperparams(args.best_hyperparams_file)
        config_info = {
            "gmm_preset": best_hyperparams["trial"]["gmm_preset"]["name"],
            "tau": best_hyperparams["trial"]["tau"],
            "alpha": best_hyperparams["trial"]["alpha"],
            "model": args.model_path,
        }

    from report import generate_report

    report_path = os.path.join(args.output_dir, "report", "report.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    generate_report(
        report_images=split.report_images,
        candidate_pool_cache=cache,
        classification_file=classification_path,
        question_file=qfile_path,
        answers_file=answers_path,
        image_dir=args.image_folder,
        output_path=report_path,
        config_info=config_info,
    )
    update_summary(args.output_dir, report_html=report_path)
    print(f"[Strategy8-U] HTML report written to {report_path} (no LVLM was loaded for this stage).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strategy 8-U end-to-end pipeline")

    p.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--image_folder", type=str, default="./data/coco/val2014")
    p.add_argument("--chair_question_file", type=str, default="./data/org_qa/chair/coco_chair.json")
    p.add_argument("--pope_question_file", type=str, default="./data/org_qa/pope/coco/coco_pope_adversarial.json")
    p.add_argument("--detr_guidance_file", type=str, default="./data/marine_qa/guidance/coco_detr_th0.95.json")
    p.add_argument("--ram_guidance_file", type=str, default="./data/marine_qa/guidance/coco_ram_th0.68.json")
    p.add_argument("--coco_annotations_path", type=str, default="./data/coco/annotations")

    p.add_argument("--output_dir", type=str, default="./output/llava2/strategy8_union")
    p.add_argument("--owlvit_model", type=str, default="google/owlvit-base-patch32")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")

    p.add_argument("--n_tune_images", type=int, default=300)
    p.add_argument("--n_report_images", type=int, default=100)
    p.add_argument("--split_seed", type=int, default=8)

    p.add_argument("--stage", type=str, default="all", choices=STAGE_CHOICES,
                    help="Which part of the pipeline to run this invocation: "
                         "'all' (default: tune-if-requested + CHAIR + POPE + report), "
                         "'tune_only' (just the hyperparameter search), "
                         "'chair' (just the CHAIR final eval, needs best_hyperparams.json), "
                         "'pope' (just the POPE final eval, needs best_hyperparams.json), "
                         "'report' (just the report -- needs the CHAIR test200 run to already "
                         "exist; does NOT load the LVLM).")

    p.add_argument("--tune", action="store_true",
                    help="Toggle: run the hyperparameter grid search (item #5). "
                         "If not set, --best_hyperparams_file must already exist "
                         "(except for --stage report).")
    p.add_argument("--max_trials", type=int, default=16)
    p.add_argument("--grid_seed", type=int, default=0)
    p.add_argument("--taus", type=_csv_floats, default=None,
                    help="Comma-separated tau values to search, e.g. '0.2,0.3,0.4,0.5'. "
                         "Defaults to hyperparam_grid.DEFAULT_TAUS.")
    p.add_argument("--alphas", type=_csv_floats, default=None,
                    help="Comma-separated alpha values to search, e.g. '0.5,0.6,0.7,0.8'. "
                         "Defaults to hyperparam_grid.DEFAULT_ALPHAS.")
    p.add_argument("--first_tau", type=float, default=0.3,
                    help="tau value of the trial guaranteed to be evaluated first (if present "
                         "in the searched tau/alpha values). Set to a value outside --taus, or "
                         "pass --first_tau -1 together with --first_alpha -1, to disable forcing.")
    p.add_argument("--first_alpha", type=float, default=0.7,
                    help="alpha value of the trial guaranteed to be evaluated first.")
    p.add_argument("--tune_learning_rate", action="store_true",
                    help="Also consider damped (lr<1.0) GMM M-step variants during GMM preset "
                         "selection. Off by default: only standard, undamped EM (lr=1.0) presets "
                         "are considered.")
    p.add_argument("--no_area_feature", action="store_true",
                    help="Exclude s_area from GMM features, using only [s_det, s_clip] (2D). "
                         "Default (flag absent): use all 3 features [s_det, s_clip, sqrt(s_area)], "
                         "all z-score normalized, which gives the best cluster quality. "
                         "Must be consistent across --tune and all --stage runs in the same output_dir.")
    p.add_argument("--force_recompute_trials", action="store_true",
                    help="Re-run GMM preset selection and every (tau, alpha) trial even if a "
                         "cached result already exists on disk. Off by default: any trial (or "
                         "the GMM selection step) whose result is already saved is REUSED, not "
                         "re-evaluated.")
    p.add_argument("--best_hyperparams_file", type=str, default=None,
                    help="Defaults to <output_dir>/best_hyperparams.json")

    p.add_argument("--conv_mode", type=str, default="vicuna_v1")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--seed", type=int, default=242)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force_recompute_pool", action="store_true")

    return p


def main():
    args = build_arg_parser().parse_args()
    if args.best_hyperparams_file is None:
        args.best_hyperparams_file = os.path.join(args.output_dir, "best_hyperparams.json")
    if args.first_tau is not None and args.first_tau < 0:
        args.first_tau = None
    if args.first_alpha is not None and args.first_alpha < 0:
        args.first_alpha = None

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import set_seed
    set_seed(args.seed)

    # ---- --stage report / set_hyperparams are lightweight: no LVLM (if the
    # cache already exists), no CHAIR evaluator, no feature extractors. ----
    if args.stage == "report":
        run_report_stage(args)
        return
    if args.stage == "set_hyperparams":
        run_set_hyperparams_stage(args)
        return

    if args.stage == "tune_only" and not args.tune:
        raise ValueError("--stage tune_only without --tune does nothing useful; pass --tune too.")
    if args.stage in ("chair", "pope", "all") and not args.tune and not os.path.exists(args.best_hyperparams_file):
        raise FileNotFoundError(
            f"--stage {args.stage} needs {args.best_hyperparams_file} to already exist "
            f"(or pass --tune to create it first)."
        )

    # ---- image universe + deterministic split ----
    with open(args.chair_question_file) as f:
        chair_questions = json.load(f)
    all_images = sorted({q["image"] for q in chair_questions})

    split_path = os.path.join(args.output_dir, "split.json")
    if os.path.exists(split_path):
        split = ImageSplit.load(split_path)
        print(f"[Strategy8-U] Reusing existing split: {split_path}")
    else:
        split = make_split(all_images, n_tune=args.n_tune_images, n_report=args.n_report_images, seed=args.split_seed)
        split.save(split_path)
        print(f"[Strategy8-U] Created new split ({len(split.tune_images)} tune / "
              f"{len(split.test_images)} test / {len(split.report_images)} report) -> {split_path}")

    # ---- load the LVLM ONCE, reused for every step below; OWL-ViT/CLIP are
    # only constructed lazily, inside ensure_candidate_pool_cache, if the
    # candidate pool cache doesn't already exist on disk ----
    model, tokenizer, processor, model_name = generate.load_strategy8_model(args.model_path)

    def _feature_extractor_factory():
        from feature_extractors import FeatureExtractor
        return FeatureExtractor(args.owlvit_model, args.clip_model, device=args.device)

    cache = ensure_candidate_pool_cache(args, model, tokenizer, processor, _feature_extractor_factory, all_images)

    # ---- CHAIR evaluator: only needed for tuning and the 'chair'/'all' stages ----
    need_chair_evaluator = args.tune or args.stage in ("chair", "all")
    chair_evaluator = None
    if need_chair_evaluator:
        from eval_chair import CHAIR
        print("[Strategy8-U] Building CHAIR evaluator (loading COCO ground-truth annotations)...")
        chair_evaluator = CHAIR(args.coco_annotations_path)

    # ---- hyperparameter search (item #5 toggle) ----
    if args.tune:
        best_hyperparams = run_hyperparameter_search(
            args, cache, split, model, tokenizer, processor, model_name, chair_evaluator,
        )
    else:
        best_hyperparams = load_best_hyperparams(args.best_hyperparams_file)
        print(f"[Strategy8-U] Loaded previously-tuned hyperparameters from {args.best_hyperparams_file}: "
              f"{best_hyperparams['trial']}")

    if args.stage == "tune_only":
        return

    gmm_params = GMMParams.from_dict(best_hyperparams["gmm_params"])
    gmm = GlobalGMM.from_params(gmm_params)
    scaler = FeatureScaler.from_dict(best_hyperparams["feature_scaler"])
    tau = best_hyperparams["trial"]["tau"]
    alpha = best_hyperparams["trial"]["alpha"]

    # ---- final evaluation: held-out 200 + full 500 ----
    if args.stage in ("chair", "all"):
        chair_test = run_final_chair_eval(
            args, cache, split.test_images, "test200", gmm, scaler, tau, alpha,
            model, tokenizer, processor, model_name, chair_evaluator,
        )
        chair_full = run_final_chair_eval(
            args, cache, split.all_images, "full500", gmm, scaler, tau, alpha,
            model, tokenizer, processor, model_name, chair_evaluator,
        )
        update_summary(
            args.output_dir,
            best_hyperparams=best_hyperparams,
            split_file=split_path,
            chair_test200=chair_test["eval_file"],
            chair_full500=chair_full["eval_file"],
        )

    if args.stage in ("pope", "all"):
        pope_test = run_final_pope_eval(
            args, cache, split.test_images, "test200", gmm, scaler, tau, alpha,
            model, tokenizer, processor, model_name,
        )
        pope_full = run_final_pope_eval(
            args, cache, split.all_images, "full500", gmm, scaler, tau, alpha,
            model, tokenizer, processor, model_name,
        )
        update_summary(
            args.output_dir,
            best_hyperparams=best_hyperparams,
            split_file=split_path,
            pope_test200=pope_test["eval_file"],
            pope_full500=pope_full["eval_file"],
        )

    print(f"[Strategy8-U] Done. Summary written to {os.path.join(args.output_dir, 'summary.json')}")

    # ---- HTML report (item #6), built from the CHAIR test-200 run's
    # artifacts -- only happens automatically as part of --stage all (which
    # just produced them above); for a standalone report from a PRIOR
    # chair run, use --stage report instead. ----
    if args.stage == "all":
        from report import generate_report

        report_path = os.path.join(args.output_dir, "report", "report.html")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        generate_report(
            report_images=split.report_images,
            candidate_pool_cache=cache,
            classification_file=chair_test["classification_file"],
            question_file=chair_test["question_file"],
            answers_file=chair_test["answers_file"],
            image_dir=args.image_folder,
            output_path=report_path,
            config_info={
                "gmm_preset": best_hyperparams["trial"]["gmm_preset"]["name"],
                "tau": tau,
                "alpha": alpha,
                "model": args.model_path,
            },
        )
        update_summary(args.output_dir, report_html=report_path)
        print(f"[Strategy8-U] HTML report written to {report_path}")


if __name__ == "__main__":
    main()
