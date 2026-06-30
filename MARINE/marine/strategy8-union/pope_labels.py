"""
pope_labels.py
==============

`eval/eval_pope.py::load_labels` expects a JSON array of
{"id": ..., "image": ..., "label": "yes"|"no"} entries, matching the
`id` field used in whatever question file the corresponding answers were
generated from. The original codebase ships a precomputed file with this
exact shape (data/marine_qa/label/pope_..._label.json), but its `id`
numbering is tied to ITS OWN (intersection-based) guidance question file,
not necessarily to whatever image subset we restrict to (the tuning/
held-out/full-500 splits from splits.py).

We instead derive the label file directly from the ORIGINAL, unmodified
POPE question file (data/org_qa/pope/coco/coco_pope_adversarial.json),
which already carries a `label` per question -- so the ids are guaranteed
to line up with whatever subset of that same file build_question_file.py
was restricted to.
"""

from __future__ import annotations

import json
from typing import List, Optional


def load_pope_questions(path: str) -> List[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        with open(path) as f:
            return [json.loads(line) for line in f]


def build_pope_label_file(
    pope_question_path: str,
    output_path: str,
    image_filter: Optional[List[str]] = None,
) -> int:
    """Writes a label file in eval/eval_pope.py's expected schema, derived
    from the original POPE question file, optionally restricted to
    `image_filter`. Returns the number of entries written."""
    questions = load_pope_questions(pope_question_path)
    image_filter_set = set(image_filter) if image_filter is not None else None

    labels = []
    for q in questions:
        img = q["image"]
        if image_filter_set is not None and img not in image_filter_set:
            continue
        labels.append({
            "id": q.get("question_id", q.get("id")),
            "image": img,
            "label": q["label"],
        })

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)
    return len(labels)
