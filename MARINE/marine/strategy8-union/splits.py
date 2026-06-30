"""
splits.py
=========

Deterministic image-level train/test split, used to satisfy item #7 of the
spec: hyperparameters (the global GMM fit + tau + alpha grid, see
hyperparam_grid.py) are chosen using 300 of the 500 CHAIR/POPE images;
the remaining 200 are held out for the final evaluation report, of which
100 are rendered in the HTML report. The full 500-image run (with the
winning hyperparameters) is stored separately again afterwards.

CHAIR and POPE in this codebase are confirmed to cover the exact same 500
COCO images (verified directly against data/org_qa/chair/coco_chair.json
and data/org_qa/pope/coco/coco_pope_adversarial.json), so a single image-id
split is reused for both benchmarks -- this also means the (expensive)
Phase I feature-extraction cache (candidate_pool.py) only has to be
computed once per image, not once per benchmark.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ImageSplit:
    all_images: List[str]
    tune_images: List[str]      # used to fit the global GMM + select hyperparameters
    test_images: List[str]      # held out, used for the final report evaluation
    report_images: List[str]    # subset of test_images rendered in the HTML report

    def to_dict(self) -> dict:
        return {
            "all_images": self.all_images,
            "tune_images": self.tune_images,
            "test_images": self.test_images,
            "report_images": self.report_images,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageSplit":
        return cls(
            all_images=d["all_images"],
            tune_images=d["tune_images"],
            test_images=d["test_images"],
            report_images=d["report_images"],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ImageSplit":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def make_split(
    all_images: Sequence[str],
    n_tune: int = 300,
    n_report: int = 100,
    seed: int = 8,
) -> ImageSplit:
    images = sorted(set(all_images))  # sort first: stable input order regardless of upstream iteration order
    if n_tune >= len(images):
        raise ValueError(f"n_tune ({n_tune}) must be smaller than the total image count ({len(images)})")

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    tune_images = sorted(shuffled[:n_tune])
    test_images = sorted(shuffled[n_tune:])

    if n_report > len(test_images):
        raise ValueError(f"n_report ({n_report}) cannot exceed the held-out test set size ({len(test_images)})")
    rng2 = random.Random(seed + 1)
    report_images = sorted(rng2.sample(test_images, n_report))

    return ImageSplit(
        all_images=images,
        tune_images=tune_images,
        test_images=test_images,
        report_images=report_images,
    )
