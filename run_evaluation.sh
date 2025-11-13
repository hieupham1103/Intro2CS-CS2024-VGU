#!/bin/bash

# Evaluation script for drone detection models
# Usage: ./run_evaluation.sh

echo "========================================"
echo "  Drone Detection Model Evaluation"
echo "========================================"
echo ""

# Configuration
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
EVAL_IOU_THRESHOLD=0.5
DEVICE="cuda"

# Create output directory
mkdir -p outputs/evaluation

echo "Starting evaluation..."
echo ""

# Evaluate YOLO RGB Model
echo "1. Evaluating YOLO RGB Model..."
python evaluate.py \
    --model-path checkpoints/rgb.pt \
    --model-type yolo \
    --data-config configs/data_RGB.yml \
    --conf-threshold $CONF_THRESHOLD \
    --iou-threshold $IOU_THRESHOLD \
    --eval-iou-threshold $EVAL_IOU_THRESHOLD \
    --device $DEVICE \
    --output-dir outputs/evaluation

echo ""
echo "2. Evaluating Multiscale RGB Model..."
python evaluate.py \
    --model-path checkpoints/rgb.pt \
    --model-type multiscale \
    --data-config configs/data_RGB.yml \
    --conf-threshold 0.2 \
    --iou-threshold 0.1 \
    --eval-iou-threshold $EVAL_IOU_THRESHOLD \
    --scales 1.0 1.2 \
    --crop-ratio 0.65 \
    --device $DEVICE \
    --output-dir outputs/evaluation

echo ""
echo "3. Evaluating YOLO IR Model..."
python evaluate.py \
    --model-path checkpoints/ir.pt \
    --model-type yolo \
    --data-config configs/data_IR.yml \
    --conf-threshold $CONF_THRESHOLD \
    --iou-threshold $IOU_THRESHOLD \
    --eval-iou-threshold $EVAL_IOU_THRESHOLD \
    --device $DEVICE \
    --output-dir outputs/evaluation

echo ""
echo "========================================"
echo "  Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved in: outputs/evaluation/"
echo ""
echo "Files:"
echo "  - yolo_evaluation_results.txt"
echo "  - multiscale_evaluation_results.txt"
echo ""
