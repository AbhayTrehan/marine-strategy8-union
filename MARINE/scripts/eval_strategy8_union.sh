#!/usr/bin/env bash
# Strategy 8-U: Two-Pass Candidate Union + Tri-State Contrastive Decoding
#
# This is a NEW script -- it does not modify scripts/eval_llava2.sh or any
# other file outside marine/strategy8-union/. It mirrors that script's
# conventions (MODEL_VERSION/SEED/BATCH_SIZE variables, PYTHONPATH setup
# for the externally-cloned LLaVA repo) so it slots into the same workflow.
#
# Usage:
#   # Step 1: one-time hyperparameter search (item #5)
#   bash scripts/eval_strategy8_union.sh --stage tune_only --tune
#
#   # Step 2: run only CHAIR (or only POPE, or only the report) once tuned
#   bash scripts/eval_strategy8_union.sh --stage chair
#   bash scripts/eval_strategy8_union.sh --stage pope
#   bash scripts/eval_strategy8_union.sh --stage report   # no LVLM needed
#
#   # Or do everything in one go (tune + CHAIR + POPE + report)
#   bash scripts/eval_strategy8_union.sh --tune
#
#   # Subsequent full runs: reuse the already-tuned hyperparameters
#   bash scripts/eval_strategy8_union.sh
#
# All flags after the script name are forwarded as-is to run_pipeline.py,
# so e.g. `bash scripts/eval_strategy8_union.sh --tune --max_trials 8` or
# `bash scripts/eval_strategy8_union.sh --stage pope` both work.

set -e

export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

MODEL_VERSION="llava-hf/llava-1.5-7b-hf"
SEED=242
BATCH_SIZE=1

OUTPUT_DIR=./output/llava2/strategy8_union

echo "Running Strategy 8-U pipeline with $MODEL_VERSION (seed=$SEED, batch_size=$BATCH_SIZE)"
echo "Output directory: $OUTPUT_DIR"

python ./marine/strategy8-union/run_pipeline.py \
    --model_path "$MODEL_VERSION" \
    --image_folder ./data/coco/val2014 \
    --chair_question_file ./data/org_qa/chair/coco_chair.json \
    --pope_question_file ./data/org_qa/pope/coco/coco_pope_adversarial.json \
    --detr_guidance_file ./data/marine_qa/guidance/coco_detr_th0.95.json \
    --ram_guidance_file ./data/marine_qa/guidance/coco_ram_th0.68.json \
    --coco_annotations_path ./data/coco/annotations \
    --output_dir "$OUTPUT_DIR" \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_new_tokens 64 \
    "$@"

echo "Done. See $OUTPUT_DIR/summary.json and $OUTPUT_DIR/report/report.html"
