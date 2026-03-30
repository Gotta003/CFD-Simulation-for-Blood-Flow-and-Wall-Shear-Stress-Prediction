#!/usr/bin/bash
#1) Setup Environment
#chmod +x setup.sh
#./setup.sh

#2) Activate Env
#source .evar_env/bin/activate

#3) Pipeline Start Dataset
set -euo pipefail
VTP_DIR="data/vtp_files"
FEATURES_CSV="outputs/features/features.csv"
OUTCOMES_CSV="data/labels/outcomes.csv"
NPZ_CHECKS_DIR="outputs/npz_checks/"
POINTCLOUD_DIR="outputs/pointclouds/"
DATASET_DIR="outputs/dataset"
SPLITS_DIR="outputs/splits"
N_POINTS=8192
STRATEGY="fps"
SEED=42

echo "Step 1/3 - Sample point clouds from .vtp meshes"
python src/datasets/samples_pointclouds.py \
	--vtp_dir "$VTP_DIR" \
	--out_dir "$POINTCLOUD_DIR" \
	--n_points "$N_POINTS" \
	--strategy "$STRATEGY" \
	--seed "$SEED"

python3 src/visualization/check_cloudpoints.py \
	--input "$POINTCLOUD_DIR" \
	--out_dir "$NPZ_CHECKS_DIR"

echo "Step 2/3 - Merge CFD features with outcomes labels"
python src/datasets/build_dataset.py \
	--features "$FEATURES_CSV" \
	--outcomes "$OUTCOMES_CSV" \
	--out_dir "$DATASET_DIR" \
	--pointcloud_dir "$POINTCLOUD_DIR"

echo "Step 3/3 - Generate 5-fold splits"
python src/data/split_dataset.py \
	--dataset "$DATASET_DIR/dataset.csv" \
	--out_dir "$SPLITS_DIR" \
	--n_folds 5 \
	--test_pct 0.20 \
	--seeed "$SEED"

echo ""
echo "=========================================================="
echo "Pipeline complete. Outputs:"
echo "Point clouds: $POINTCLOUD_DIR/"
echo "Dataset CSV: $DATASET_DIR/dataset.csv"
echo "Feature list: $DATASET_DIR/feature_columns.txt"
echo "Label Summary: $DATASET_DIR/label_summary.txt"
echo "Splits: $SPLITS_DIR/"
echo "=========================================================="
