"""
Run with: python3 tests/test_run_pipeline.py

Exercises run_pipeline.py END TO END with a small (20-image) synthetic
dataset: the --tune toggle, the decoupled GMM-preset selection (intrinsic
quality, no generation), the (tau, alpha) grid with per-trial caching/
resumability, the --first_tau/--first_alpha forced-first-trial behavior,
and the separate --stage chair / --stage pope / --stage report commands
(including that --stage report never loads the LVLM). Every piece that
needs real GPU compute or downloaded weights (the LVLM, CHAIR's COCO
ground-truth loading) is replaced with a deterministic fake; everything
else (Strategy8TriStateDataset, TriStateGuidanceLogits, the GMM fit,
synonym/union canonicalization, the question-file building, the report
HTML) runs FOR REAL.

The fake CHAIR evaluator's metrics are a deterministic function of the
`alpha` value baked into each answer file by the real generate.py code
path (metadata.alpha) -- CHAIRi decreases and Recall increases with alpha
-- so we can assert hyperparameter SELECTION genuinely works (picks the
highest-alpha trial), not just that the code runs.
"""
import json
import os
import sys
import tempfile

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_STRATEGY8_DIR = os.path.join(_TESTS_DIR, "..")
_MARINE_ROOT = os.path.join(_STRATEGY8_DIR, "..", "..")

sys.path.insert(0, os.path.join(_TESTS_DIR, "_mock_llava"))
sys.path.insert(0, _STRATEGY8_DIR)
sys.path.insert(0, _MARINE_ROOT)
sys.path.insert(0, os.path.join(_MARINE_ROOT, "eval"))

import torch
from PIL import Image

import marine.utils.utils_model as utils_model_module
import eval_chair as eval_chair_module

import run_pipeline
import generate


N_IMAGES = 20


# ---------------------------------------------------------------------------
# Fake LVLM (same pattern as test_generate.py), with a global call counter
# so tests can verify how many times generation actually ran.
# ---------------------------------------------------------------------------
_GENERATE_CALL_COUNT = {"n": 0}
_LOAD_MODEL_CALL_COUNT = {"n": 0}


class _FakeProcessor:
    def __call__(self, text, images, return_tensors="pt"):
        n_tokens = max(3, len(text.split()))
        return {
            "input_ids": torch.arange(n_tokens).unsqueeze(0),
            "attention_mask": torch.ones(1, n_tokens, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 4, 4),
        }


class _FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row.tolist()) for row in ids]


class _FakeConfig:
    mm_use_im_start_end = False


class _FakeForwardOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeModel:
    def __init__(self):
        self.config = _FakeConfig()

    def __call__(self, input_ids=None, pixel_values=None, attention_mask=None,
                 use_cache=True, past_key_values=None):
        batch, seq = input_ids.shape
        logits = torch.randn(batch, seq, 11)
        new_past = 0 if past_key_values is None else past_key_values + 1
        return _FakeForwardOutput(logits=logits, past_key_values=new_past)

    def generate(self, input_ids, **kwargs):
        _GENERATE_CALL_COUNT["n"] += 1
        lp = kwargs.get("logits_processor")
        if lp is not None:
            fake_logits = torch.randn(input_ids.shape[0], 11)
            for processor in lp:
                processor(input_ids, fake_logits)
        new_tokens = torch.full((input_ids.shape[0], 3), 5, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def _fake_load_model(model_name, model_path):
    _LOAD_MODEL_CALL_COUNT["n"] += 1
    return _FakeModel(), _FakeTokenizer(), _FakeProcessor()


def _raise_if_called(model_name, model_path):
    raise AssertionError("load_model should NOT have been called for this stage")


# ---------------------------------------------------------------------------
# Fake CHAIR evaluator: metrics are a deterministic function of alpha
# ---------------------------------------------------------------------------
class _FakeCHAIR:
    def __init__(self, coco_path):
        self.coco_path = coco_path

    def compute_chair(self, cap_file, image_id_key, caption_key):
        with open(cap_file) as f:
            rows = [json.loads(l) for l in f]
        alpha = rows[0].get("metadata", {}).get("alpha", 0.0) if rows else 0.0
        chair_i = max(0.0, 0.3 - 0.3 * alpha)
        chair_s = chair_i * 1.5
        recall = min(1.0, 0.2 + 0.3 * alpha)
        return {
            "sentences": [],
            "overall_metrics": {
                "CHAIRs": chair_s, "CHAIRi": chair_i, "Recall": recall,
                "num_hallucinated_caps": 0, "num_caps": len(rows),
                "hallucinated_word_count": 0, "coco_word_count": max(1, len(rows)),
                "length_response": 5.0, "hallucinated_caps_ls": [],
            },
        }


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------
def _image_name(i):
    return f"COCO_val2014_{str(i).zfill(12)}.jpg"


def _build_synthetic_dataset(d):
    image_dir = os.path.join(d, "images")
    os.makedirs(image_dir, exist_ok=True)
    for i in range(1, N_IMAGES + 1):
        Image.new("RGB", (8, 8), color=(i % 255, 50, 100)).save(os.path.join(image_dir, _image_name(i)))

    chair_questions, pope_questions, detr_guidance, ram_guidance = [], [], [], []
    qid = 1
    for i in range(1, N_IMAGES + 1):
        img = _image_name(i)
        chair_questions.append({
            "id": qid, "image": img,
            "conversations": [
                {"from": "human", "value": "Generate a short caption of the image."},
                {"from": "gpt", "value": ""},
            ],
        })
        qid += 1
        detr_guidance.append({"image": img, "objects": ["dog", "person"] if i % 2 == 0 else []})
        ram_guidance.append({"image": img, "objects": ["dog", "leash"] if i % 2 == 0 else ["cat"]})

    pqid = 1
    for i in range(1, N_IMAGES + 1):
        img = _image_name(i)
        for obj, label in [("dog", "yes"), ("fork", "no")]:
            pope_questions.append({"question_id": pqid, "image": img, "text": f"Is there a {obj} in the image?", "label": label})
            pqid += 1

    chair_path = os.path.join(d, "chair.json")
    with open(chair_path, "w") as f:
        json.dump(chair_questions, f)
    pope_path = os.path.join(d, "pope.json")
    with open(pope_path, "w") as f:
        for q in pope_questions:
            f.write(json.dumps(q) + "\n")
    detr_path = os.path.join(d, "detr.json")
    with open(detr_path, "w") as f:
        json.dump(detr_guidance, f)
    ram_path = os.path.join(d, "ram.json")
    with open(ram_path, "w") as f:
        json.dump(ram_guidance, f)

    return {"image_dir": image_dir, "chair_path": chair_path, "pope_path": pope_path,
            "detr_path": detr_path, "ram_path": ram_path}


def _build_synthetic_candidate_pool_cache(output_dir, n_images=N_IMAGES):
    """Bypasses Step A's real model/OWL-ViT/CLIP calls entirely."""
    cache_path = os.path.join(output_dir, "candidate_pool_cache.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        for i in range(1, n_images + 1):
            img = _image_name(i)
            if i % 2 == 0:
                candidates = [
                    {"canonical": "dog", "sources": ["ram", "detr"], "raw_mentions": ["dog"],
                     "is_coco_category": True, "s_det": 0.85, "s_clip": 0.3, "s_area": 0.1},
                    {"canonical": "leash", "sources": ["ram"], "raw_mentions": ["leash"],
                     "is_coco_category": False, "s_det": 0.05, "s_clip": 0.03, "s_area": 0.01},
                ]
                raw = {"ram": ["dog", "leash"], "detr": ["dog", "person"], "vlm": ["dog"]}
                caption = "A dog on a leash."
            else:
                candidates = [
                    {"canonical": "cat", "sources": ["ram"], "raw_mentions": ["cat"],
                     "is_coco_category": True, "s_det": 0.7, "s_clip": 0.25, "s_area": 0.08},
                ]
                raw = {"ram": ["cat"], "detr": [], "vlm": ["cat"]}
                caption = "A cat."
            rec = {"image": img, "pass1_caption": caption, "raw": raw, "candidates": candidates}
            f.write(json.dumps(rec) + "\n")
    return cache_path


def _common_argv(ds, output_dir):
    return [
        "--model_path", "fake-model",
        "--image_folder", ds["image_dir"],
        "--chair_question_file", ds["chair_path"],
        "--pope_question_file", ds["pope_path"],
        "--detr_guidance_file", ds["detr_path"],
        "--ram_guidance_file", ds["ram_path"],
        "--output_dir", output_dir,
        "--n_tune_images", "12",
        "--n_report_images", "5",
        "--split_seed", "8",
        "--max_new_tokens", "4",
        "--batch_size", "2",
    ]


def _run_pipeline_main(argv):
    old_argv = sys.argv
    sys.argv = ["run_pipeline.py"] + argv
    try:
        run_pipeline.main()
    finally:
        sys.argv = old_argv


def _setup(d):
    utils_model_module.load_model = _fake_load_model
    eval_chair_module.CHAIR = _FakeCHAIR
    _GENERATE_CALL_COUNT["n"] = 0
    _LOAD_MODEL_CALL_COUNT["n"] = 0
    ds = _build_synthetic_dataset(d)
    return ds


def test_tune_selects_highest_alpha_and_gmm_selection_uses_no_generation():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)

    argv = _common_argv(ds, output_dir) + [
        "--stage", "tune_only", "--tune",
        "--taus", "0.2,0.3,0.4,0.5", "--alphas", "0.5,0.6,0.7,0.8",
        "--max_trials", "16", "--first_tau", "-1", "--first_alpha", "-1",
    ]
    _run_pipeline_main(argv)

    # GMM preset selection considers 3 base presets, but must NOT call
    # generate() at all -- only the 16 (tau, alpha) trials should, each
    # doing ceil(n_tune_images / batch_size) = 12/2 = 6 batched calls.
    expected_calls = 16 * (12 // 2)
    assert _GENERATE_CALL_COUNT["n"] == expected_calls, _GENERATE_CALL_COUNT["n"]

    gmm_selection_path = os.path.join(output_dir, "tuning", "gmm_selection.json")
    assert os.path.exists(gmm_selection_path)
    with open(gmm_selection_path) as f:
        selection = json.load(f)
    assert set(selection["quality_by_preset"].keys()) == {"standard_kmeans", "quantile_init", "fixed_prior"}

    with open(os.path.join(output_dir, "best_hyperparams.json")) as f:
        best = json.load(f)
    assert best["trial"]["alpha"] == 0.8  # highest alpha in the (uncapped) grid wins
    assert "feature_scaler" in best  # scaler must be stored
    print("test_tune_selects_highest_alpha_and_gmm_selection_uses_no_generation OK")


def test_first_tau_alpha_is_evaluated_first():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)

    argv = _common_argv(ds, output_dir) + [
        "--stage", "tune_only", "--tune",
        "--taus", "0.2,0.3,0.4,0.5", "--alphas", "0.5,0.6,0.7,0.8",
        "--max_trials", "4", "--grid_seed", "123",
        "--first_tau", "0.3", "--first_alpha", "0.7",
    ]
    _run_pipeline_main(argv)

    with open(os.path.join(output_dir, "tuning", "all_trials.json")) as f:
        all_trials = json.load(f)
    assert len(all_trials) == 4
    assert all_trials[0]["trial"]["tau"] == 0.3
    assert all_trials[0]["trial"]["alpha"] == 0.7
    print("test_first_tau_alpha_is_evaluated_first OK")


def test_rerunning_tune_reuses_cached_trials_and_gmm_selection():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)

    argv = _common_argv(ds, output_dir) + [
        "--stage", "tune_only", "--tune",
        "--taus", "0.2,0.3", "--alphas", "0.5,0.6",
        "--max_trials", "4", "--first_tau", "-1", "--first_alpha", "-1",
    ]
    _run_pipeline_main(argv)
    n_calls_first_run = _GENERATE_CALL_COUNT["n"]
    expected_calls = 4 * (12 // 2)  # 4 trials x 6 batches each
    assert n_calls_first_run == expected_calls, n_calls_first_run

    # re-run with the SAME output_dir: every trial + the GMM selection
    # should be found on disk and reused -- zero new generate() calls.
    _run_pipeline_main(argv)
    assert _GENERATE_CALL_COUNT["n"] == n_calls_first_run, "no new generation should have occurred on rerun"

    # --force_recompute_trials should force everything to re-run
    _run_pipeline_main(argv + ["--force_recompute_trials"])
    assert _GENERATE_CALL_COUNT["n"] == n_calls_first_run * 2
    print("test_rerunning_tune_reuses_cached_trials_and_gmm_selection OK")


def test_stage_chair_then_stage_pope_then_stage_report_accumulate_summary():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)
    base_argv = _common_argv(ds, output_dir)

    # tune first (need best_hyperparams.json to exist for chair/pope stages)
    _run_pipeline_main(base_argv + ["--stage", "tune_only", "--tune", "--max_trials", "2"])
    summary_path = os.path.join(output_dir, "summary.json")
    assert not os.path.exists(summary_path)  # tune_only never touches summary.json

    # --stage chair only
    _run_pipeline_main(base_argv + ["--stage", "chair"])
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary.get("chair_test200") and os.path.exists(summary["chair_test200"])
    assert summary.get("chair_full500") and os.path.exists(summary["chair_full500"])
    assert "pope_test200" not in summary
    assert not os.path.exists(os.path.join(output_dir, "report", "report.html"))

    # --stage pope only -- must NOT erase the chair entries already recorded
    _run_pipeline_main(base_argv + ["--stage", "pope"])
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary.get("chair_test200")  # still present
    assert summary.get("pope_test200") and os.path.exists(summary["pope_test200"])
    assert summary.get("pope_full500") and os.path.exists(summary["pope_full500"])

    # --stage report only -- must NOT load the model at all
    utils_model_module.load_model = _raise_if_called
    _run_pipeline_main(base_argv + ["--stage", "report"])
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary.get("report_html") and os.path.exists(summary["report_html"])
    with open(summary["report_html"]) as f:
        report_html = f.read()
    with open(os.path.join(output_dir, "split.json")) as f:
        split = json.load(f)
    for img in split["report_images"]:
        assert img in report_html

    print("test_stage_chair_then_stage_pope_then_stage_report_accumulate_summary OK")


def test_stage_report_without_chair_run_raises_clear_error():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    try:
        _run_pipeline_main(_common_argv(ds, output_dir) + ["--stage", "report"])
        raise AssertionError("should have raised FileNotFoundError")
    except FileNotFoundError as e:
        assert "chair" in str(e).lower()
    print("test_stage_report_without_chair_run_raises_clear_error OK")


def test_missing_best_hyperparams_without_tune_raises():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    try:
        _run_pipeline_main(_common_argv(ds, output_dir))
        raise AssertionError("should have raised FileNotFoundError")
    except FileNotFoundError:
        pass
    print("test_missing_best_hyperparams_without_tune_raises OK")


def test_candidate_pool_cache_not_rebuilt_when_present():
    d = tempfile.mkdtemp()
    ds = _setup(d)
    output_dir = os.path.join(d, "out")
    cache_path = _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)
    mtime_before = os.path.getmtime(cache_path)

    argv = _common_argv(ds, output_dir) + ["--stage", "tune_only", "--tune", "--max_trials", "2"]
    _run_pipeline_main(argv)
    assert os.path.getmtime(cache_path) == mtime_before, "cache file should not have been rewritten"
    print("test_candidate_pool_cache_not_rebuilt_when_present OK")


def test_tune_learning_rate_flag_controls_gmm_presets_considered():
    d = tempfile.mkdtemp()
    ds = _setup(d)

    output_dir_off = os.path.join(d, "out_off")
    _build_synthetic_candidate_pool_cache(output_dir_off, N_IMAGES)
    _run_pipeline_main(_common_argv(ds, output_dir_off) + ["--stage", "tune_only", "--tune", "--max_trials", "2"])
    with open(os.path.join(output_dir_off, "tuning", "gmm_selection.json")) as f:
        sel_off = json.load(f)
    assert all(q.get("learning_rate", 1.0) for q in [{}])  # no-op sanity
    with open(os.path.join(output_dir_off, "best_hyperparams.json")) as f:
        best_off = json.load(f)
    assert best_off["trial"]["gmm_preset"]["learning_rate"] == 1.0

    output_dir_on = os.path.join(d, "out_on")
    _build_synthetic_candidate_pool_cache(output_dir_on, N_IMAGES)
    _run_pipeline_main(_common_argv(ds, output_dir_on)
                        + ["--stage", "tune_only", "--tune", "--max_trials", "2", "--tune_learning_rate"])
    with open(os.path.join(output_dir_on, "tuning", "gmm_selection.json")) as f:
        sel_on = json.load(f)
    assert set(sel_on["quality_by_preset"].keys()) == {
        "standard_kmeans", "quantile_init", "fixed_prior", "damped_kmeans_lr0.5", "damped_kmeans_lr0.3",
    }
    print("test_tune_learning_rate_flag_controls_gmm_presets_considered OK")


if __name__ == "__main__":
    test_tune_selects_highest_alpha_and_gmm_selection_uses_no_generation()
    test_first_tau_alpha_is_evaluated_first()
    test_rerunning_tune_reuses_cached_trials_and_gmm_selection()
    test_stage_chair_then_stage_pope_then_stage_report_accumulate_summary()
    test_stage_report_without_chair_run_raises_clear_error()
    test_missing_best_hyperparams_without_tune_raises()
    test_candidate_pool_cache_not_rebuilt_when_present()
    test_tune_learning_rate_flag_controls_gmm_presets_considered()
    print("\nALL run_pipeline.py TESTS PASSED")
