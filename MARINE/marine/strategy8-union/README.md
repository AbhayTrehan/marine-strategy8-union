# Strategy 8-U: Two-Pass Candidate Union + Tri-State Contrastive Decoding

Implementation of `Strategy8_Union_contrastive.pdf` on top of the MARINE
codebase. This README documents what's here, how it fits together, how to
run it, and the design decisions made along the way.

**Zero modifications to the original codebase.** Every file in this
directory is new. The only other new file anywhere in the repo is
`scripts/eval_strategy8_union.sh`. Nothing under `marine/utils/`, `eval/`,
or `marine/generate_llava2.py` was touched -- this package only ever
*imports* those modules (synonym tables, the CHAIR/POPE evaluators, the
LVLM loader, dataset/chunking helpers).

## What the strategy does

1. **Candidate union** (`O_init = O_det u O_vlm`): for each image, the
   precomputed DETR/RAM++ tag lists are unioned with nouns extracted from
   the LVLM's own unguided first-pass caption.
2. **3D feature extraction**: every candidate gets a real
   `[s_det, s_clip, s_area]` vector from a zero-shot OWL-ViT detector query
   ("a photo of a {object}") and CLIP image-text similarity.
3. **Phase I, offline sorter**: a single 2-component Gaussian Mixture Model
   is fit (via Expectation-Maximization) on these feature vectors and used
   to split each image's candidates into `O_pos` (verified) and `O_neg`
   (statistically flagged as hallucinated).
4. **Phase II, tri-state contrastive decoding**: three parallel logit
   branches (unconditioned, positive-conditioned, negative-conditioned)
   are blended at every decoding step:
   `z_final = (1-alpha)*z_ung + alpha*(z_pos - z_neg)`.

## Important deviations from a literal reading of the algorithm box, confirmed with the user

* **The GMM is fit GLOBALLY, not per-image.** The paper's Algorithm 1 box
  fits one GMM per image, which is numerically fragile (an image might
  propose 0-5 candidates -- nowhere near enough to fit a stable 3D,
  2-component, full-covariance mixture) and gives every image its own
  ungrounded notion of "high" vs. "low" evidence. Per explicit
  instruction, we instead pool candidate feature vectors across a
  *tuning* set of images, fit one GMM on the pool, freeze its parameters,
  and apply only the E-step to classify candidates in any image (tuning,
  held-out, or the full dataset). See `gmm.py`'s module docstring.
* **Tri-state decoding** uses the corrected spec
  (`Strategy8_Union_contrastive.pdf`), not the earlier two-branch draft:
  three forward passes per decoding step (unconditioned, c_pos, c_neg),
  blended via Eq. 20. This needed a genuinely new `LogitsProcessor`
  (`tristate_logits.py`) rather than a 2-branch reuse of the original
  codebase's `GuidanceLogits`.
* **O_det is built from the already-precomputed DETR(th=0.95)/RAM++(th=0.68)
  tag files** in `data/marine_qa/guidance/`, per the user's explicit
  choice, rather than re-running those detectors.

## File-by-file map

| File | Role |
|---|---|
| `synonyms.py` | Union-merge canonicalization: curated COCO synonym table (reused from `eval/find_intersection.py`) + WordNet noun-synonym clustering for everything outside that table. |
| `text_objects.py` | Extracts candidate noun mentions from the VLM's free-text Pass-1 caption, mirroring CHAIR's own tokenization/double-word machinery (`eval/eval_chair.py`) without needing COCO annotations loaded. |
| `feature_extractors.py` | Real OWL-ViT (`s_det`, `s_area`) and CLIP (`s_clip`) scorers, batched per image. |
| `gmm.py` | From-scratch 2-component multivariate GMM EM (validated against `sklearn.mixture.GaussianMixture`), with a damped/configurable M-step ("learning rate"), 3 init strategies, and a frozen-params E-step-only `responsibility_positive()` for applying a fitted GMM to new images. |
| `prompts.py` | Builds `c_ung` / `c_pos` / `c_neg` text per Section 4.1. |
| `dataset.py` | `Strategy8TriStateDataset` + collate: tokenizes all three branches per question, mirroring `marine/utils/utils_dataset.py`'s prompt-building pattern. |
| `tristate_logits.py` | `TriStateGuidanceLogits`: the 3-branch `LogitsProcessor` implementing Eq. 17-20, with independent KV caches per guidance branch. |
| `candidate_pool.py` | **Step A**: builds `O_init` + extracts features for every image. Run once, cached to disk, independent of every tunable hyperparameter. |
| `fit_gmm.py` | **Step B**: pools features from a set of "fitting" images and fits the global GMM. Pure numpy, no GPU. |
| `build_question_file.py` | **Step C**: applies a *frozen* GMM + tau threshold to classify candidates, then builds the tri-state question file for a specific benchmark (CHAIR or POPE) and image subset. Pure numpy, no GPU. |
| `generate.py` | **Step D**: the actual tri-state generation loop (the GPU-expensive step). Exposes `load_strategy8_model()` / `run_generation()` separately so the model can be loaded once and reused across many trials. |
| `hyperparam_grid.py` | The (bounded) hyperparameter search space + F1-from-CHAIR scoring (item #5). |
| `splits.py` | Deterministic 300/200/100 image split (item #7). |
| `pope_labels.py` | Derives a POPE label file from the original question file, scoped to whatever image subset is being evaluated. |
| `run_pipeline.py` | Top-level orchestrator: Step A once -> optional `--tune` grid search -> final CHAIR+POPE evaluation on held-out 200 + full 500 -> report data handoff. |
| `report.py` | HTML report generator (item #6), built entirely from already-produced artifacts (no model calls). |
| `tests/` | Unit + integration tests (see "How this was validated" below). |

## Setup

Everything here reuses the original codebase's dependencies
(`transformers`, `torch`, `nltk`, `scikit-learn`) plus two that the
original codebase already implicitly requires without listing in
`requirements.txt` (`textblob`, used by `eval/eval_chair.py`; `shortuuid`,
used by `marine/generate_llava2.py`) -- no new packages beyond those:

```bash
pip install textblob shortuuid   # if not already present
python -c "import nltk; [nltk.download(p) for p in \
  ['punkt','punkt_tab','wordnet','omw-1.4','stopwords']]"
```

OWL-ViT and CLIP are downloaded automatically on first use via
`transformers` (`google/owlvit-base-patch32`, `openai/clip-vit-base-patch32`
by default; override with `--owlvit_model`/`--clip_model`).

## Quick start (step by step)

```bash
# 0. Merge this package into your existing MARINE/ checkout (skip if you
#    already unzipped it there): copy marine/strategy8-union/ and
#    scripts/eval_strategy8_union.sh into your repo, alongside the
#    untouched original files.
cd MARINE/

# 1. One-time setup: extra deps + NLTK data (see "Setup" below for why
#    these specific packages -- they're already implicit dependencies of
#    the original codebase, nothing new beyond it).
pip install textblob shortuuid
python -c "import nltk; [nltk.download(p) for p in \
  ['punkt','punkt_tab','wordnet','omw-1.4','stopwords']]"

# 2. Make sure you already have, per the main README: LLaVA cloned with
#    PYTHONPATH set, and COCO val2014 images + annotations downloaded
#    under ./data/coco/. (Same prerequisites as eval_llava2.sh -- nothing
#    extra needed for Strategy 8-U here.)

# 3. SMOKE TEST FIRST -- cheap, fast, catches environment issues (wrong
#    transformers version, missing OWL-ViT download access, etc.) before
#    you commit GPU time to a real run. --max_trials 2 means only 2
#    generation passes over the 300 tuning images instead of 12.
bash scripts/eval_strategy8_union.sh --tune --max_trials 2

# 4. If step 3 looks right (check ./output/llava2/strategy8_union/summary.json
#    and report/report.html), run the real tuning pass. This is the
#    expensive step -- see "Compute cost note" below.
bash scripts/eval_strategy8_union.sh --tune --max_trials 12

# 5. From now on, reuse the tuned hyperparameters without re-tuning -- this
#    just re-runs the final CHAIR/POPE/report evaluation (e.g. after you
#    change the model, or just to regenerate the report):
bash scripts/eval_strategy8_union.sh
```

Everything after the script name is forwarded straight to
`run_pipeline.py`, so any flag below can be appended to either command,
e.g. `bash scripts/eval_strategy8_union.sh --tune --max_trials 20 --tune_learning_rate`.

Useful flags:
* `--max_trials N` -- cap on the hyperparameter grid (default 12).
* `--tune_learning_rate` -- also search damped (lr<1.0) GMM M-step
  variants. **Off by default**: the grid only fits the GMM with standard,
  undamped EM (lr=1.0). See "On the GMM 'learning rate'" below.
* `--n_tune_images` / `--n_report_images` -- default 300 / 100, per item #7.
* `--skip_chair`, `--skip_pope`, `--skip_report`, `--skip_final_eval`.
* `--force_recompute_pool` -- rebuild the Step A cache even if present.

### On the GMM "learning rate"

Standard GMM EM has a closed-form M-step (Eq. 9-12) -- there's no literal
learning rate to set. What `gmm.py` implements instead is a *damping
factor* on that closed-form update:

```
theta_new = (1 - lr) * theta_old + lr * theta_closed_form
```

`lr = 1.0` is exactly standard, undamped EM. `lr < 1.0` slows/smooths the
trajectory, which mainly helps if the pooled tuning-image feature set is
small or noisy. It is **not tuned by default** -- `select_gmm_presets()`
in `hyperparam_grid.py` only includes `lr=1.0` presets unless you pass
`--tune_learning_rate`, in which case two damped variants (`lr=0.5`,
`lr=0.3`) are added to the search. If you want a specific fixed damping
value without searching anything, edit `BASE_GMM_PRESETS` in
`hyperparam_grid.py` directly (or pass `gmm_presets=[...]` to
`build_grid()` if calling it from your own script).

Output lands under `--output_dir` (default
`./output/llava2/strategy8_union/`):

```
candidate_pool_cache.jsonl     # Step A, all images
split.json                     # tune/test/report image lists
best_hyperparams.json          # winning (gmm_preset, tau, alpha) + frozen GMM params
tuning/                        # per-trial question files, answers, CHAIR eval, all_trials.json
final/
  chair_answers_test200.jsonl / chair_eval_test200.json
  chair_answers_full500.jsonl / chair_eval_full500.json
  pope_answers_test200.jsonl  / pope_eval_test200.json
  pope_answers_full500.jsonl  / pope_eval_full500.json
report/report.html             # item #6
summary.json
```

## Compute cost note

Hyperparameter tuning runs real generation for every trial in the grid on
the 300 tuning images; only the GMM-fit dimensions (learning rate,
init strategy, max_iters) are cheap to vary (pure numpy over the cached
features from Step A) -- `tau` and `alpha` each require a fresh generation
pass. `--max_trials` bounds this; the default grid (3 lr=1.0 GMM presets x
3 taus x 3 alphas = 27, capped to 12 by default; +2 more presets =
45 if `--tune_learning_rate` is also passed) is a deliberately small,
curated search rather than an exhaustive one. The final evaluation phase
always regenerates the full 500-image runs from scratch (rather than
splicing together tuning-phase partial results) for simplicity and
correctness.

## How this was validated

This was developed in a sandboxed environment with **no GPU and no access
to the HuggingFace Hub or COCO image servers** -- i.e. the real LLaVA,
OWL-ViT, and CLIP weights, and the COCO val2014 images themselves, could
not be downloaded or run here. Everything that doesn't need them was
written as real, runnable code and exercised with real unit/integration
tests in `tests/` (run with `python3 tests/test_<module>.py`, or all of
them via the loop in the next section):

* **`gmm.py`** -- the from-scratch EM implementation is checked against
  `sklearn.mixture.GaussianMixture` on synthetic data (means/log-likelihood
  match to ~1e-3), plus damping, all 3 init strategies, frozen-apply to
  unseen data, serialization, and degenerate-input edge cases.
* **`synonyms.py` / `text_objects.py`** -- exercised against real NLTK
  WordNet/stopwords/tokenizer data (downloaded into this sandbox) with
  real example captions and RAM/DETR-style tag lists.
* **`tristate_logits.py`** -- validated against a hand-rolled but *real*
  causal self-attention model with an actual KV cache (`tests/test_tristate_logits.py`),
  checking that the incremental/cached computation is numerically
  identical to a from-scratch recomputation at every decoding step, and
  that the Eq. 20 blending formula matches a manual reference computation.
* **`dataset.py` / `generate.py`** -- exercised against a synthetic
  processor/tokenizer and a fake-but-structurally-faithful model (real
  forward-pass call pattern, real KV cache threading) via a minimal mock
  `llava` package (`tests/_mock_llava/`) standing in for the externally
  cloned LLaVA repo, since that's a manual git-clone dependency not
  available here either.
* **`run_pipeline.py`** -- a full 20-image synthetic end-to-end run
  (`tests/test_run_pipeline.py`) exercising the real `--tune` toggle, the
  real hyperparameter grid, real GMM fitting/freezing, real question-file
  building, and a real HTML report -- with a deterministic fake CHAIR
  evaluator whose metrics are a function of `alpha`, which lets the test
  assert that hyperparameter *selection* actually works (picks the
  genuinely best trial), not just that the code runs.
* **`feature_extractors.py`** -- the real OWL-ViT/CLIP classes cannot be
  instantiated here (no weights), but the post-processing math
  (`owlvit_postprocess`, `clip_cosine_similarities`) they're built on is
  factored out and tested directly with hand-constructed tensors matching
  those models' real output contracts.

What was **not**, and could not be, executed in this sandbox: an actual
forward pass through real LLaVA/OWL-ViT/CLIP weights, and anything
requiring the real COCO val2014 images or annotation files (which are not
included in this repo due to size and must be downloaded per the main
`README.md`'s instructions). The first real run of
`bash scripts/eval_strategy8_union.sh --tune` should therefore be watched
for early errors (e.g. transformers-version-specific API differences in
`feature_extractors.py` or `marine/utils/utils_model.py::load_model`,
which this package does not modify) before launching a long tuning run.

To run the whole test suite:

```bash
cd marine/strategy8-union
for f in tests/test_*.py; do echo "=== $f ==="; python3 "$f" || break; done
```
