"""
prompts.py
==========

Builds the three contextual prompts used by Phase II Tri-State Contrastive
Decoding (Strategy8_Union_contrastive.pdf, Section 4.1):

    1. c_ung: the plain task query, unchanged ("Generate a short caption
       of the image.", "Is there a keyboard in the image?", ...).
    2. c_pos: the query enriched with the verified object set O_pos --
       "Focusing on the visible objects in this image: [O_pos]. {query}"
    3. c_neg: the query enriched with the statistically-flagged
       hallucinated set O_neg, using the SAME affirmative phrasing as
       c_pos (the paper is explicit this must not be a negation) --
       "Focusing on the visible objects in this image: [O_neg]. {query}"

Object-list formatting mirrors eval/prompt_template.py::PromptTemplate.obj_ls2str
(",", "and", oxford-comma-less join) so the resulting prompts read the same
way the original MARINE guidance prompts do.

Empty-list handling: if O_pos (or O_neg) is empty for a given image, the
corresponding prompt degrades gracefully to the plain query (no dangling
"Focusing on the visible objects in this image: . ..."), matching the
fallback behavior already used by PromptTemplate.generate_prompt for an
empty object_list_str.
"""

from __future__ import annotations

from typing import List, Sequence


GUIDANCE_PREFIX = "Focusing on the visible objects in this image"


def objects_to_string(objects: Sequence[str]) -> str:
    objects = [o for o in objects if o]
    if len(objects) == 0:
        return ""
    if len(objects) == 1:
        return objects[0]
    if len(objects) == 2:
        return f"{objects[0]} and {objects[1]}"
    return ", ".join(objects[:-1]) + ", and " + objects[-1]


def build_unconditional_prompt(query: str) -> str:
    return query


def build_object_guided_prompt(query: str, objects: Sequence[str]) -> str:
    """Section 4.1: 'Focusing on the visible objects in this image: [O].
    {query}' -- used identically for both c_pos (O=O_pos) and c_neg
    (O=O_neg); the wording is intentionally affirmative in both cases."""
    obj_str = objects_to_string(list(objects))
    if not obj_str:
        return query
    q = query[0].lower() + query[1:] if query else query
    return f"{GUIDANCE_PREFIX}: {obj_str}. {q}"


def build_tristate_prompts(query: str, o_pos: Sequence[str], o_neg: Sequence[str]):
    """Returns (c_ung, c_pos, c_neg) for one (image, query) pair."""
    c_ung = build_unconditional_prompt(query)
    c_pos = build_object_guided_prompt(query, o_pos)
    c_neg = build_object_guided_prompt(query, o_neg)
    return c_ung, c_pos, c_neg
