"""Run with: python3 tests/test_report.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

from report import generate_report, build_candidate_table, get_base64_image


def _make_image(d, name):
    path = os.path.join(d, name)
    Image.new("RGB", (8, 8), color=(100, 150, 200)).save(path)
    return path


def _make_artifacts(d):
    image_dir = d
    _make_image(d, "img0.jpg")
    _make_image(d, "img1.jpg")

    candidate_pool_cache = {
        "img0.jpg": {
            "image": "img0.jpg",
            "pass1_caption": "A dog is sitting near a fork on the table.",
            "raw": {"ram": ["dog", "table"], "detr": ["dog"], "vlm": ["dog", "fork", "table"]},
            "candidates": [
                {"canonical": "dog", "sources": ["ram", "detr", "vlm"], "raw_mentions": ["dog"],
                 "is_coco_category": True, "s_det": 0.82, "s_clip": 0.31, "s_area": 0.12},
                {"canonical": "dining table", "sources": ["ram", "vlm"], "raw_mentions": ["table"],
                 "is_coco_category": True, "s_det": 0.7, "s_clip": 0.25, "s_area": 0.3},
                {"canonical": "fork", "sources": ["vlm"], "raw_mentions": ["fork"],
                 "is_coco_category": True, "s_det": 0.03, "s_clip": 0.02, "s_area": 0.01},
            ],
        },
        "img1.jpg": {
            "image": "img1.jpg",
            "pass1_caption": "A cat.",
            "raw": {"ram": [], "detr": [], "vlm": ["cat"]},
            "candidates": [
                {"canonical": "cat", "sources": ["vlm"], "raw_mentions": ["cat"],
                 "is_coco_category": True, "s_det": 0.6, "s_clip": 0.2, "s_area": 0.1},
            ],
        },
    }

    classification = {
        "img0.jpg": {
            "o_pos": ["dog", "dining table"],
            "o_neg": ["fork"],
            "responsibilities": {"dog": 0.97, "dining table": 0.88, "fork": 0.02},
        },
        "img1.jpg": {
            "o_pos": ["cat"],
            "o_neg": [],
            "responsibilities": {"cat": 0.91},
        },
    }
    classification_path = os.path.join(d, "classification.json")
    with open(classification_path, "w") as f:
        json.dump(classification, f)

    questions = [
        {
            "id": 1, "image": "img0.jpg",
            "conversations": [
                {"from": "human", "value": "Generate a short caption of the image."},
                {"from": "gpt", "value": ""},
                {"from": "guidance_pos", "value": "Focusing on the visible objects in this image: dog and dining table. generate a short caption of the image."},
                {"from": "guidance_neg", "value": "Focusing on the visible objects in this image: fork. generate a short caption of the image."},
            ],
        },
        {
            "id": 2, "image": "img1.jpg",
            "conversations": [
                {"from": "human", "value": "Generate a short caption of the image."},
                {"from": "gpt", "value": ""},
                {"from": "guidance_pos", "value": "Focusing on the visible objects in this image: cat. generate a short caption of the image."},
                {"from": "guidance_neg", "value": "Generate a short caption of the image."},
            ],
        },
    ]
    question_path = os.path.join(d, "questions.json")
    with open(question_path, "w") as f:
        json.dump(questions, f)

    answers = [
        {"question_id": 1, "image_id": "img0.jpg", "text": "A dog sits near a table."},
        {"question_id": 2, "image_id": "img1.jpg", "text": "A cat."},
    ]
    answers_path = os.path.join(d, "answers.jsonl")
    with open(answers_path, "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")

    return candidate_pool_cache, classification_path, question_path, answers_path, image_dir


def test_generate_report_basic():
    d = tempfile.mkdtemp()
    cache, cls_path, q_path, ans_path, image_dir = _make_artifacts(d)
    out_path = os.path.join(d, "report.html")

    result_path = generate_report(
        report_images=["img0.jpg", "img1.jpg"],
        candidate_pool_cache=cache,
        classification_file=cls_path,
        question_file=q_path,
        answers_file=ans_path,
        image_dir=image_dir,
        output_path=out_path,
        config_info={"tau": 0.5, "alpha": 0.6},
    )
    assert result_path == out_path
    assert os.path.exists(out_path)
    with open(out_path) as f:
        html_content = f.read()

    # both images should be present
    assert "img0.jpg" in html_content
    assert "img1.jpg" in html_content
    # (a) base response
    assert "A dog is sitting near a fork on the table." in html_content
    # (h) final response
    assert "A dog sits near a table." in html_content
    # (g) prompts
    assert "Focusing on the visible objects in this image: dog and dining table" in html_content
    assert "Focusing on the visible objects in this image: fork" in html_content
    # (f) gamma values
    assert "0.9700" in html_content
    # config banner
    assert "0.5" in html_content and "0.6" in html_content
    print("test_generate_report_basic OK")


def test_generate_report_skips_missing_images_gracefully():
    d = tempfile.mkdtemp()
    cache, cls_path, q_path, ans_path, image_dir = _make_artifacts(d)
    out_path = os.path.join(d, "report.html")

    generate_report(
        report_images=["img0.jpg", "does_not_exist.jpg"],
        candidate_pool_cache=cache,
        classification_file=cls_path,
        question_file=q_path,
        answers_file=ans_path,
        image_dir=image_dir,
        output_path=out_path,
    )
    with open(out_path) as f:
        content = f.read()
    assert "showing 1 of 2 requested images" in content
    print("test_generate_report_skips_missing_images_gracefully OK")


def test_build_candidate_table_orders_positive_first():
    candidates = [
        {"canonical": "fork", "sources": ["vlm"], "raw_mentions": ["fork"], "s_det": 0.02, "s_clip": 0.01, "s_area": 0.01},
        {"canonical": "dog", "sources": ["ram"], "raw_mentions": ["dog"], "s_det": 0.9, "s_clip": 0.3, "s_area": 0.1},
    ]
    responsibilities = {"fork": 0.05, "dog": 0.95}
    table = build_candidate_table(candidates, responsibilities, o_pos_set={"dog"}, o_neg_set={"fork"})
    assert table.index("dog") < table.index("fork")
    assert "POSITIVE" in table and "HALLUCINATED" in table
    print("test_build_candidate_table_orders_positive_first OK")


def test_build_candidate_table_empty():
    table = build_candidate_table([], {}, set(), set())
    assert "No candidate objects" in table
    print("test_build_candidate_table_empty OK")


def test_get_base64_image_missing_file_returns_empty_string():
    assert get_base64_image("/nonexistent/path/x.jpg") == ""
    print("test_get_base64_image_missing_file_returns_empty_string OK")


def test_html_escaping_prevents_injection():
    d = tempfile.mkdtemp()
    cache, cls_path, q_path, ans_path, image_dir = _make_artifacts(d)
    # inject something that looks like a tag into a caption
    cache["img0.jpg"]["pass1_caption"] = "<script>alert(1)</script> a dog"
    out_path = os.path.join(d, "report.html")
    generate_report(["img0.jpg"], cache, cls_path, q_path, ans_path, image_dir, out_path)
    with open(out_path) as f:
        content = f.read()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
    print("test_html_escaping_prevents_injection OK")


if __name__ == "__main__":
    test_generate_report_basic()
    test_generate_report_skips_missing_images_gracefully()
    test_build_candidate_table_orders_positive_first()
    test_build_candidate_table_empty()
    test_get_base64_image_missing_file_returns_empty_string()
    test_html_escaping_prevents_injection()
    print("\nALL report.py TESTS PASSED")
