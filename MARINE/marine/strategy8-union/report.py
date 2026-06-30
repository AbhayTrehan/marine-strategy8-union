"""
report.py
=========

Builds the HTML visual report requested in item #6 of the spec. For each
of the (up to 100) report images, shows:
  (a) base unguided VLM response (the Phase I Pass-1 caption)
  (b) objects mentioned by RAM++
  (c) objects mentioned by DETR
  (d) objects mentioned by the base VLM itself (Pass-1 noun extraction)
  (e) the final O_init canonical union after synonym mapping (synonyms.py),
      with provenance (which source(s) contributed each canonical object)
  (f) the positive-cluster responsibility gamma_i (Eq. 8, frozen GMM) for
      every candidate, and the resulting O_pos / O_neg split (Eq. 15-16)
  (g) the actual c_pos / c_neg guidance prompts used for this image's
      query (Section 4.1)
  (h) the final response after tri-state contrastive decoding (Eq. 20)

Everything here is assembled from artifacts run_pipeline.py has ALREADY
produced (the candidate pool cache, the per-image classification file, the
strategy8 question file, and the CHAIR test-200 answers file) -- this
module makes no model calls of its own and can be re-run cheaply to
restyle the report without regenerating anything.
"""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Dict, List


def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _badge_list(items, css_class: str) -> str:
    if not items:
        return '<span class="empty-note">none</span>'
    return "".join(f'<span class="obj-badge {css_class}">{_esc(o)}</span>' for o in items)


def build_candidate_table(candidates: List[dict], responsibilities: Dict[str, float],
                           o_pos_set: set, o_neg_set: set) -> str:
    if not candidates:
        return "<p class='no-objects'>No candidate objects in O_init for this image.</p>"

    rows = ""
    # show positive candidates first (sorted by gamma desc), then negative
    def _gamma_of(c):
        g = responsibilities.get(c["canonical"])
        return g if g is not None else -1.0

    ordered = sorted(candidates, key=lambda c: (-1 if c["canonical"] in o_pos_set else 0, -_gamma_of(c)))

    for c in ordered:
        canonical = c["canonical"]
        gamma = responsibilities.get(canonical)
        is_pos = canonical in o_pos_set
        row_class = "pos-row" if is_pos else "neg-row"
        badge = (
            '<span class="badge badge-pos">POSITIVE</span>' if is_pos
            else '<span class="badge badge-neg">HALLUCINATED</span>'
        )
        sources_str = ", ".join(sorted(c.get("sources", [])))
        raw_str = ", ".join(c.get("raw_mentions", []))
        gamma_str = f"{gamma:.4f}" if gamma is not None else "N/A"
        rows += f"""
        <tr class="{row_class}">
            <td><strong>{_esc(canonical)}</strong></td>
            <td>{_esc(sources_str)}</td>
            <td class="raw-col">{_esc(raw_str)}</td>
            <td>{c.get('s_det', 0.0):.3f}</td>
            <td>{c.get('s_clip', 0.0):.3f}</td>
            <td>{c.get('s_area', 0.0):.3f}</td>
            <td class="gamma-col">{gamma_str}</td>
            <td>{badge}</td>
        </tr>
        """
    return f"""
    <table class="score-table">
        <thead>
        <tr>
            <th>Canonical Object</th>
            <th>Sources</th>
            <th>Raw Mentions</th>
            <th>s_det (OWL-ViT)</th>
            <th>s_clip (CLIP)</th>
            <th>s_area</th>
            <th>&gamma; (Positive resp.)</th>
            <th>Status</th>
        </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def build_image_card(
    idx: int,
    img: str,
    pool: dict,
    question_entry: dict,
    final_response: str,
    classification: dict,
    image_dir: str,
) -> str:
    convs = {c["from"]: c["value"] for c in question_entry["conversations"]}
    c_pos_text = convs.get("guidance_pos", "")
    c_neg_text = convs.get("guidance_neg", "")

    o_pos_set = set(classification.get("o_pos", []))
    o_neg_set = set(classification.get("o_neg", []))
    responsibilities = classification.get("responsibilities", {})

    candidates = pool.get("candidates", [])
    raw = pool.get("raw", {"ram": [], "detr": [], "vlm": []})

    candidate_table_html = build_candidate_table(candidates, responsibilities, o_pos_set, o_neg_set)
    img_b64 = get_base64_image(os.path.join(image_dir, img))

    return f"""  <div class="card">
    <div class="card-header">
      <span>Image {idx} &nbsp;|&nbsp; {_esc(img)}</span>
      <span>O_init: {len(candidates)} &nbsp;|&nbsp;
            <span style="color:#166534;font-weight:700;">O_pos: {len(o_pos_set)}</span> &nbsp;|&nbsp;
            <span style="color:#991b1b;font-weight:700;">O_neg: {len(o_neg_set)}</span></span>
    </div>
    <div class="card-body">
      <div class="image-container">
        <img src="{img_b64}" alt="{_esc(img)}">
      </div>
      <div class="content">

        <div class="section-title">(a) Base LVLM response &mdash; unguided Pass-1 caption (y&#8201;<sup>(1)</sup>)</div>
        <div class="text-box original">{_esc(pool.get('pass1_caption', ''))}</div>

        <div class="section-title">(b)&ndash;(d) Raw object mentions by source</div>
        <div class="source-row"><span class="source-label src-ram">RAM++</span> {_badge_list(raw.get('ram', []), 'badge-ram')}</div>
        <div class="source-row"><span class="source-label src-detr">DETR</span> {_badge_list(raw.get('detr', []), 'badge-detr')}</div>
        <div class="source-row"><span class="source-label src-vlm">VLM (Pass-1)</span> {_badge_list(raw.get('vlm', []), 'badge-vlm')}</div>

        <div class="section-title">(e)&ndash;(f) O_init: union after synonym mapping, with &gamma; and the O_pos / O_neg split</div>
        {candidate_table_html}

        <div class="section-title">(g) Guidance prompts actually used: c_pos / c_neg</div>
        <div class="text-box prompt-pos"><span class="prompt-label">c_pos</span>{_esc(c_pos_text)}</div>
        <div class="text-box prompt-neg"><span class="prompt-label">c_neg</span>{_esc(c_neg_text)}</div>

        <div class="section-title">(h) Final response after tri-state contrastive decoding</div>
        <div class="text-box corrected">{_esc(final_response)}</div>

      </div>
    </div>
  </div>
"""


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f0f2f5;
    color: #2d2d2d;
    margin: 0;
    padding: 20px;
  }}
  h1 {{ text-align: center; color: #1a1a2e; margin-bottom: 6px; font-size: 24px; }}
  .subtitle {{ text-align: center; color: #555; margin-bottom: 24px; font-size: 14px; }}
  .container {{ max-width: 1320px; margin: 0 auto; }}

  .banner {{
    background: #1a1a2e; color: white; padding: 20px 28px; border-radius: 10px;
    margin-bottom: 28px; box-shadow: 0 4px 14px rgba(0,0,0,.18);
  }}
  .banner h2 {{ margin: 0 0 8px 0; font-size: 16px; color: #b0b8d8; letter-spacing: .3px; }}
  .banner .meta {{ font-size: 13px; color: #9ea8cc; }}
  .banner .meta b {{ color: #e8ecff; }}

  .card {{
    background: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.09);
    margin-bottom: 28px; overflow: hidden;
  }}
  .card-header {{
    background: #f7f8fa; padding: 12px 20px; border-bottom: 1px solid #e5e7eb;
    font-size: 13px; color: #666; display: flex; justify-content: space-between; align-items: center;
  }}
  .card-body {{ display: flex; flex-wrap: wrap; }}
  .image-container {{
    flex: 0 0 320px; background: #111; display: flex; align-items: center;
    justify-content: center; padding: 12px; min-height: 240px;
  }}
  .image-container img {{ max-width: 100%; max-height: 300px; object-fit: contain; border-radius: 4px; }}
  .content {{ flex: 1; padding: 20px 24px; min-width: 0; }}

  .section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px;
    color: #888; margin: 18px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #eee;
  }}
  .section-title:first-child {{ margin-top: 0; }}

  .text-box {{
    padding: 12px 14px; border-left: 4px solid #3b82f6; border-radius: 0 6px 6px 0;
    background: #fafbff; font-size: 14.5px; line-height: 1.55;
  }}
  .text-box.original {{ border-left-color: #6b7280; }}
  .text-box.corrected {{ border-left-color: #22c55e; background: #f0fdf4; font-weight: 500; }}
  .text-box.prompt-pos {{ border-left-color: #16a34a; background: #f0fdf4; margin-bottom: 8px; }}
  .text-box.prompt-neg {{ border-left-color: #dc2626; background: #fef2f2; }}
  .prompt-label {{
    display: inline-block; font-family: monospace; font-weight: 700; font-size: 12px;
    color: #555; margin-right: 8px; background: #eee; padding: 1px 6px; border-radius: 4px;
  }}

  .source-row {{ margin: 4px 0; font-size: 13.5px; line-height: 1.9; }}
  .source-label {{
    display: inline-block; min-width: 84px; font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: .3px; color: #555; margin-right: 6px;
  }}
  .obj-badge {{
    display: inline-block; padding: 2px 9px; margin: 2px 3px 2px 0; border-radius: 12px;
    font-size: 12px; font-family: monospace; border: 1px solid transparent;
  }}
  .badge-ram {{ background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }}
  .badge-detr {{ background: #fdf4ff; color: #a21caf; border-color: #f5d0fe; }}
  .badge-vlm {{ background: #fff7ed; color: #c2410c; border-color: #fed7aa; }}
  .empty-note {{ color: #9ca3af; font-style: italic; font-size: 12.5px; }}

  .score-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; font-family: monospace; }}
  .score-table thead th {{
    background: #f1f5f9; color: #374151; font-weight: 700; padding: 8px 8px;
    text-align: center; border: 1px solid #e2e8f0; white-space: nowrap;
  }}
  .score-table td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: center; vertical-align: middle; }}
  .score-table .raw-col {{ text-align: left; max-width: 220px; }}
  .score-table .gamma-col {{ font-weight: 700; color: #1d4ed8; }}
  .pos-row {{ background: #f7fef7; }}
  .neg-row {{ background: #fff5f5; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 10.5px;
    font-weight: 700; letter-spacing: .2px; white-space: nowrap;
  }}
  .badge-pos {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
  .badge-neg {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}

  .no-objects {{ color: #9ca3af; font-style: italic; font-size: 13px; margin-top: 10px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Strategy 8-U: Two-Pass Candidate Union + Tri-State Contrastive Decoding</h1>
  <p class="subtitle">Visual evaluation report &middot; showing {n_shown} of {n_requested} requested images</p>

  <div class="banner">
    <h2>Pipeline configuration used for this report</h2>
    <div class="meta">{config_html}</div>
  </div>

{cards}
</div>
</body>
</html>
"""


def generate_report(
    report_images: List[str],
    candidate_pool_cache: Dict[str, dict],
    classification_file: str,
    question_file: str,
    answers_file: str,
    image_dir: str,
    output_path: str,
    config_info: dict = None,
    title: str = "Strategy 8-U Visual Report",
) -> str:
    """Assembles the HTML report from already-produced pipeline artifacts.
    Returns the output_path for convenience."""
    classification = _load_json(classification_file)
    questions = _load_json(question_file)
    answers = _load_jsonl(answers_file)

    question_by_image = {q["image"]: q for q in questions}
    answer_by_qid = {a["question_id"]: a for a in answers}

    cards_html = []
    n_shown = 0
    for img in report_images:
        pool = candidate_pool_cache.get(img)
        q = question_by_image.get(img)
        if pool is None or q is None:
            continue
        ans = answer_by_qid.get(q["id"])
        final_response = ans["text"] if ans else "N/A (no generation found for this image/question)"
        cls = classification.get(img, {"o_pos": [], "o_neg": [], "responsibilities": {}})

        n_shown += 1
        cards_html.append(build_image_card(n_shown, img, pool, q, final_response, cls, image_dir))

    config_info = config_info or {}
    config_bits = " &nbsp;|&nbsp; ".join(f"<b>{_esc(k)}</b>: {_esc(v)}" for k, v in config_info.items())
    if not config_bits:
        config_bits = "(no config metadata supplied)"

    html_out = _PAGE_TEMPLATE.format(
        title=title,
        n_shown=n_shown,
        n_requested=len(report_images),
        config_html=config_bits,
        cards="\n".join(cards_html),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    return output_path
