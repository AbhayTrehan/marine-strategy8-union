"""
Run with: python3 tests/test_run_pipeline.py

This is the most important test in the suite: it exercises run_pipeline.py
END TO END -- the --tune toggle, the hyperparameter grid search, picking
the winning trial, freezing its GMM, and the final CHAIR+POPE+report
evaluation on the held-out/full-500 splits -- using a small (20-image)
synthetic dataset. Every piece that needs real GPU compute or downloaded
weights (the LVLM, CHAIR's COCO ground-truth loading) is replaced with a
deterministic fake; everything else (Strategy8TriStateDataset,
TriStateGuidanceLogits, the GMM fit, synonym/union canonicalization, the
question-file building, the report HTML) runs FOR REAL.

The fake CHAIR evaluator's metrics are a deterministic function of the
`alpha` value baked into each answer file by the real generate.py code
path (metadata.alpha) -- CHAIRi decreases and Recall increases with alpha
-- so we can assert run_hyperparameter_search actually picks the
highest-alpha trial available in its (possibly down-sampled) grid, i.e.
that hyperparameter SELECTION genuinely works, not just that the code
runs without crashing.
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
# Fake LVLM (same pattern as test_generate.py)
# ---------------------------------------------------------------------------
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
        lp = kwargs.get("logits_processor")
        if lp is not None:
            fake_logits = torch.randn(input_ids.shape[0], 11)
            for processor in lp:
                processor(input_ids, fake_logits)
        new_tokens = torch.full((input_ids.shape[0], 3), 5, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def _fake_load_model(model_name, model_path):
    return _FakeModel(), _FakeTokenizer(), _FakeProcessor()


# ---------------------------------------------------------------------------
# Fake CHAIR evaluator: metrics are a deterministic function of alpha
# ---------------------------------------------------------------------------
class _FakeCHAIR:
    def __init__(self, coco_path):
        self.coco_path = coco_path  # not actually used

    def compute_chair(self, cap_file, image_id_key, caption_key):
        with open(cap_file) as f:
            rows = [json.loads(l) for l in f]
        alpha = rows[0].get("metadata", {}).get("alpha", 0.0) if rows else 0.0
        chair_i = max(0.0, 0.3 - 0.3 * alpha)   # improves (lower) with alpha
        chair_s = chair_i * 1.5
        recall = min(1.0, 0.2 + 0.3 * alpha)     # improves (higher) with alpha
        return {
            "sentences": [],
            "overall_metrics": {
                "CHAIRs": chair_s,
                "CHAIRi": chair_i,
                "Recall": recall,
                "num_hallucinated_caps": 0,
                "num_caps": len(rows),
                "hallucinated_word_count": 0,
                "coco_word_count": max(1, len(rows)),
                "length_response": 5.0,
                "hallucinated_caps_ls": [],
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

    chair_questions = []
    pope_questions = []
    detr_guidance = []
    ram_guidance = []
    qid = 1
    for i in range(1, N_IMAGES + 1):
        img = _image_name(i)
        chair_questions.append({
            "id": qid,
            "image": img,
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
            pope_questions.append({
                "question_id": pqid, "image": img,
                "text": f"Is there a {obj} in the image?", "label": label,
            })
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

    return {
        "image_dir": image_dir,
        "chair_path": chair_path,
        "pope_path": pope_path,
        "detr_path": detr_path,
        "ram_path": ram_path,
    }


def _build_synthetic_candidate_pool_cache(output_dir, n_images=N_IMAGES):
    """Bypasses Step A's real model/OWL-ViT/CLIP calls entirely by writing
    the cache file run_pipeline.ensure_candidate_pool_cache will find
    already on disk (and therefore reuse without recomputation)."""
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


def _run_pipeline_main(argv):
    old_argv = sys.argv
    sys.argv = ["run_pipeline.py"] + argv
    try:
        run_pipeline.main()
    finally:
        sys.argv = old_argv


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


def test_tune_then_reuse_end_to_end():
    utils_model_module.load_model = _fake_load_model
    eval_chair_module.CHAIR = _FakeCHAIR

    d = tempfile.mkdtemp()
    ds = _build_synthetic_dataset(d)
    output_dir = os.path.join(d, "out")
    _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)

    common_argv = _common_argv(ds, output_dir)

    # ---- phase 1: --tune ----
    _run_pipeline_main(common_argv + ["--tune", "--max_trials", "4", "--grid_seed", "1"])

    best_path = os.path.join(output_dir, "best_hyperparams.json")
    assert os.path.exists(best_path)
    with open(best_path) as f:
        best = json.load(f)

    all_trials_path = os.path.join(output_dir, "tuning", "all_trials.json")
    with open(all_trials_path) as f:
        all_trials = json.load(f)
    assert len(all_trials) == 4

    # the fake CHAIR's metrics strictly improve with alpha -> the winning
    # trial must be the one with the maximum alpha among the 4 sampled
    max_alpha_in_grid = max(t["trial"]["alpha"] for t in all_trials)
    assert best["trial"]["alpha"] == max_alpha_in_grid
    best_f1 = max(t["f1"] for t in all_trials)
    assert abs(best["tuning_result"]["f1"] - best_f1) < 1e-9

    # ---- phase 2: --skip_final_eval should stop right after tuning ----
    output_dir2 = os.path.join(d, "out2")
    _build_synthetic_candidate_pool_cache(output_dir2, N_IMAGES)
    _run_pipeline_main(_common_argv(ds, output_dir2) + ["--tune", "--max_trials", "2", "--grid_seed", "1", "--skip_final_eval"])
    assert os.path.exists(os.path.join(output_dir2, "best_hyperparams.json"))
    assert not os.path.exists(os.path.join(output_dir2, "summary.json"))

    # ---- phase 3: re-run WITHOUT --tune, reusing best_hyperparams.json ----
    final_output_dir = output_dir  # reuse phase-1's dir (already has best_hyperparams.json + cache + split)
    _run_pipeline_main(common_argv)  # no --tune this time

    summary_path = os.path.join(final_output_dir, "summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)

    for key in ["chair_test200", "chair_full500", "pope_test200", "pope_full500"]:
        assert summary[key] is not None and os.path.exists(summary[key])

    with open(summary["chair_full500"]) as f:
        chair_full_metrics = json.load(f)
    assert chair_full_metrics["num_caps"] == N_IMAGES  # full-500 stand-in: full N_IMAGES

    with open(summary["chair_test200"]) as f:
        chair_test_metrics = json.load(f)
    split_path = os.path.join(final_output_dir, "split.json")
    with open(split_path) as f:
        split = json.load(f)
    assert chair_test_metrics["num_caps"] == len(split["test_images"])

    report_path = os.path.join(final_output_dir, "report", "report.html")
    assert os.path.exists(report_path)
    with open(report_path) as f:
        report_html = f.read()
    assert "showing 5 of 5 requested images" in report_html
    for img in split["report_images"]:
        assert img in report_html

    print("test_tune_then_reuse_end_to_end OK")


def test_missing_best_hyperparams_without_tune_raises():
    utils_model_module.load_model = _fake_load_model
    eval_chair_module.CHAIR = _FakeCHAIR

    d = tempfile.mkdtemp()
    ds = _build_synthetic_dataset(d)
    output_dir = os.path.join(d, "out")

    argv = _common_argv(ds, output_dir)
    try:
        _run_pipeline_main(argv)
        raise AssertionError("should have raised FileNotFoundError")
    except FileNotFoundError:
        pass
    print("test_missing_best_hyperparams_without_tune_raises OK")


def test_candidate_pool_cache_not_rebuilt_when_present():
    """Confirms ensure_candidate_pool_cache reuses an existing cache file
    rather than rebuilding it (and therefore never needs the lazy
    FeatureExtractor factory to actually be called)."""
    utils_model_module.load_model = _fake_load_model
    eval_chair_module.CHAIR = _FakeCHAIR

    d = tempfile.mkdtemp()
    ds = _build_synthetic_dataset(d)
    output_dir = os.path.join(d, "out")
    cache_path = _build_synthetic_candidate_pool_cache(output_dir, N_IMAGES)
    mtime_before = os.path.getmtime(cache_path)

    argv = _common_argv(ds, output_dir) + ["--tune", "--max_trials", "2", "--skip_final_eval"]
    _run_pipeline_main(argv)
    mtime_after = os.path.getmtime(cache_path)
    assert mtime_before == mtime_after, "cache file should not have been rewritten"
    print("test_candidate_pool_cache_not_rebuilt_when_present OK")


def test_tune_learning_rate_flag_controls_damped_presets():
    utils_model_module.load_model = _fake_load_model
    eval_chair_module.CHAIR = _FakeCHAIR

    d = tempfile.mkdtemp()
    ds = _build_synthetic_dataset(d)

    # default (flag absent): no damped (lr<1.0) presets should ever appear
    output_dir_off = os.path.join(d, "out_off")
    _build_synthetic_candidate_pool_cache(output_dir_off, N_IMAGES)
    _run_pipeline_main(_common_argv(ds, output_dir_off) + ["--tune", "--max_trials", "8", "--skip_final_eval"])
    with open(os.path.join(output_dir_off, "tuning", "all_trials.json")) as f:
        trials_off = json.load(f)
    assert all(t["trial"]["gmm_preset"]["learning_rate"] == 1.0 for t in trials_off)

    # with --tune_learning_rate: damped presets are eligible to be sampled
    # (use a large max_trials to make it overwhelmingly likely at least one
    # damped preset gets sampled into this small grid)
    output_dir_on = os.path.join(d, "out_on")
    _build_synthetic_candidate_pool_cache(output_dir_on, N_IMAGES)
    _run_pipeline_main(_common_argv(ds, output_dir_on)
                        + ["--tune", "--max_trials", "30", "--tune_learning_rate", "--skip_final_eval"])
    with open(os.path.join(output_dir_on, "tuning", "all_trials.json")) as f:
        trials_on = json.load(f)
    assert any(t["trial"]["gmm_preset"]["learning_rate"] < 1.0 for t in trials_on)
    print("test_tune_learning_rate_flag_controls_damped_presets OK")


if __name__ == "__main__":
    test_tune_then_reuse_end_to_end()
    test_missing_best_hyperparams_without_tune_raises()
    test_candidate_pool_cache_not_rebuilt_when_present()
    test_tune_learning_rate_flag_controls_damped_presets()
    print("\nALL run_pipeline.py TESTS PASSED")
