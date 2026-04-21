#!/usr/bin/bash
#1) Setup Environment
chmod +x setup.sh
./setup.sh

#2) Activate Env
source .evar_env/bin/activate

#3) Pipeline Start Dataset
set -euo pipefail
VTP_DIR="data/vtp_files"
FEATURES_CSV="outputs/features/features.csv"
OUTCOMES_CSV="data/labels/outcomes.csv"
NPZ_CHECKS_DIR="outputs/npz_checks/"
POINTCLOUD_DIR="outputs/pointclouds/"
DATASET_DIR="outputs/dataset"
SPLITS_DIR="outputs/splits"
SLICER_BIN="/opt/Slicer-5.10.0-linux-amd64/Slicer" 
MORPHO_DIR="data/morpho"
FEATURES_DIR="outputs/features"
MESHES_DIR="../simulation_db"
N_POINTS=16384
STRATEGY="fps"
SEED=42

echo "Step 1/4 - Sample point clouds from .vtp meshes"
python src/datasets/samples_pointclouds.py \
	--vtp_dir "$VTP_DIR" \
	--out_dir "$POINTCLOUD_DIR" \
	--n_points "$N_POINTS" \
	--strategy "$STRATEGY" \
	--seed "$SEED"

python3 src/visualization/check_cloudpoints.py \
	--input "$POINTCLOUD_DIR" \
	--out_dir "$NPZ_CHECKS_DIR"

echo "Step 2/4 - Extract morphological and radiomics features"

echo "Extracting CT and Mesh features and saving in /outputs/dataset/alignment_audit.csv"
python src/extraction/audit_alignment.py

command -v xvfb-run >/dev/null 2>&1 || { echo "xvfb not found. Installing..."; sudo apt-get update && sudo apt-get install -y xvfb;}

export PYTHONPATH=""
export PYTHONHOME=""
export PYTHONUSERBASE=""
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONUSERBASE

mkdir -p "$MORPHO_DIR"
shopt -s globstar
rm -rf "$FEATURES_DIR/morpho_unified_metrics.csv"

for patient_vtp in "$VTP_DIR"/**/*.vtp; do
	patient_folder=$(basename "$(dirname "$patient_vtp")")
	digits=$(echo "$patient_folder" | tr -dc '0-9')
	patient_id=$(printf "%03d" "$((10#$digits))")
	echo "Processing $patient_id in $patient_folder..."
	xvfb-run --auto-servernum --server-args="-screen 0 1280x1024x24" \
		env -i HOME="$HOME" DISPLAY="$DISPLAY" PATH="$PATH" \
		$SLICER_BIN \
		--python-script src/extraction/morpho_extraction_slicer.py \
		--patient_id "$patient_id" \
		--db_path "$MESHES_DIR" \
		--out_dir "$MORPHO_DIR"

	python src/extraction/metrics_computation.py \
		--patientid "$patient_id" \
		--in_folder "$MORPHO_DIR" \
		--out_folder "$FEATURES_DIR"
done


echo "Step 3/4 - Merge CFD features with outcomes labels"
python src/datasets/build_dataset.py \
	--features "$FEATURES_CSV" \
	--morpho "$FEATURES_DIR/morpho_unified_metrics.csv" \
	--outcomes "$OUTCOMES_CSV" \
	--out_dir "$DATASET_DIR" \
	--pointcloud_dir "$POINTCLOUD_DIR"

echo "Step 4/4 - Generate 5-fold splits"
python src/datasets/split_dataset.py \
	--dataset "$DATASET_DIR/dataset.csv" \
	--out_dir "$SPLITS_DIR" \
	--n_folds 5 \
	--test_pct 0.20 \
	--seed "$SEED"

echo ""
echo "=========================================================="
echo "Pipeline complete. Outputs:"
echo "Point clouds: $POINTCLOUD_DIR/"
echo "Dataset CSV: $DATASET_DIR/dataset.csv"
echo "Feature list: $DATASET_DIR/feature_columns.txt"
echo "Label Summary: $DATASET_DIR/label_summary.txt"
echo "Splits: $SPLITS_DIR/"
echo "=========================================================="
