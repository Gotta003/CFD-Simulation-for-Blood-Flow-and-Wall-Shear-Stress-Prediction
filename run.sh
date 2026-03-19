#!/usr/bin/bash
#1) Setup Environment
chmod +x setup.sh
./setup.sh

#2) Activate Env
source .evar_env/bin/activate

#3) Extract features
python src/extraction/extract_features.py \
	--input data/vtp_files/ \
	--out outputs/features/features.csv

# 4) Visualize mesh heatmaps
VTP_DIR="data/vtp_files"
VIZ_SCRIPT="src/visualization/visualize.py"
OUT="outputs/mesh_heatmaps"
for vtp_file in "$VTP_DIR"/*.vtp; do
	[ -e "$vtp_file" ] || continue
	echo "Processing: $vtp_file"

	python "$VIZ_SCRIPT" \
		--vtp "$vtp_file" \
		--all_fields
	filename=$(basename $vtp_file)
	prefix="${filename%%_*}"
	dest_dir="$OUT/$prefix"
	mkdir -p "$dest_dir"
	if ls "$OUT"/*.png >/dev/null 2>&1; then
		for img in "$OUT"/*.png; do
			echo "Moving $img to $dest_dir"
			mv -f "$img" "$dest_dir/"
		done
	else
		echo "Warning: No .png generated for $vtp_file"
	fi
done

# 5) Add patients to .csv files empty
python add_patients.py --min 1 --max 120
python patients_management.py

#XGBOOST
#echo "Training XGBoost"
#python -m src.models.train \
#	--model xgboost \
#	--features outputs/features/features.csv \
#	--labels data/labels/outcomes.csv \
#	--out outputs/models/xgboost \
#	--cv 5

#python patients_management.py