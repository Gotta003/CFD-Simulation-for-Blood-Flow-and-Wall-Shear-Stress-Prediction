import os
import argparse 
import pandas as pd
import numpy as np

def extract_3d_stats(volume_path):
    """Parses Slicer SegmentStatistics for Volume and Surface Area."""
    stats = {"Volume": 0, "SurfaceArea": 0}
    if not os.path.exists(volume_path):
        return stats
    
    v_df = pd.read_csv(volume_path)
    # Mapping specific Slicer headers 
    col_vol = "LabelmapSegmentStatisticsPlugin.volume_mm3"
    col_surf = "ClosedSurfaceSegmentStatisticsPlugin.surface_mm2"
    
    if col_vol in v_df.columns:
        stats["Volume"] = v_df[col_vol].iloc[0]
    if col_surf in v_df.columns:
        stats["SurfaceArea"] = v_df[col_surf].iloc[0]
    return stats

def main():
    parser = argparse.ArgumentParser(description="Full Morphological Feature Extraction")
    parser.add_argument("--patientid", required=True)
    parser.add_argument("--in_folder", required=True, help=r"Path to pz{ID} folders")
    parser.add_argument("--out_folder", required=True)
    args = parser.parse_args()

    base_path = os.path.join(args.in_folder, f"pz{args.patientid}")
    try:
        df_axial = pd.read_csv(os.path.join(base_path, "parameters_S.csv"))
        df_coronal = pd.read_csv(os.path.join(base_path, "parameters_A.csv"))
    except FileNotFoundError:
        print(f"[ERROR] Missing CSVs for pz{args.patientid}")
        return
    
    valid_df = df_axial[df_axial["Max Diameter (mm)"] > 10].copy()
    if valid_df.empty: return
    p1 = valid_df.iloc[0]
    idx_max = valid_df["Max Diameter (mm)"].idxmax()
    p_max = valid_df.loc[idx_max]
    baseline_diam = valid_df.iloc[0:5]["Max Diameter (mm)"].mean()
    p2_candidates = valid_df[valid_df['Max Diameter (mm)'] > (baseline_diam * 1.1)]
    p2 = p2_candidates.iloc[0] if not p2_candidates.empty else p1
    p3 = valid_df.iloc[-1]
    d4 = p3["Max Diameter (mm)"] * 0.55  
    d5 = p3["Max Diameter (mm)"] * 0.50
    l_curvilinear = abs(p3['Length (mm)'] - p1['Length (mm)'])
    # Straight length L' 
    p1_pos = np.array([p1['Cx'], p1['Cy'], p1['Slice Index']])
    p3_pos = np.array([p3['Cx'], p3['Cy'], p3['Slice Index']])
    l_straight = np.linalg.norm(p3_pos - p1_pos)
    deformation_ratio = p_max['Rmajor (mm)'] / p_max['Rminor (mm)'] if p_max['Rminor (mm)'] > 0 else 1.0
    saccular_index = p_max['Max Diameter (mm)'] / p_max['Perimeter (mm)'] * np.pi if p_max['Perimeter (mm)'] > 0 else 0
    stats_3d = extract_3d_stats(os.path.join(base_path, "volume.csv"))
    unified_row = {
        "patient_id": args.patientid,
        "D1": round(p1['Max Diameter (mm)'], 2),
        "D2": round(p2['Max Diameter (mm)'], 2),
        "D3": round(p3['Max Diameter (mm)'], 2),
        "D4": round(d4, 2), 
        "D5": round(d5, 2),
        "Dmax": round(p_max['Max Diameter (mm)'], 2),
        "Proximal_Neck_Length": round(abs(p2['Length (mm)'] - p1['Length (mm)']), 2),
        "L_Curvilinear": round(l_curvilinear, 2),
        "L_Straight": round(l_straight, 2),
        "Tortuosity": round(l_curvilinear / l_straight, 3) if l_straight > 0 else 1.0,
        "Deformation_Ratio": round(deformation_ratio, 3),
        "Saccularization_Index": round(saccular_index, 3),
        "AAA_Volume_mm3": round(stats_3d["Volume"], 1),
        "AAA_SurfaceArea_mm2": round(stats_3d["SurfaceArea"], 1)
    }

    # Save to unified CSV
    output_path = os.path.join(args.out_folder, "morpho_unified_metrics.csv")
    final_df = pd.DataFrame([unified_row])
    final_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    print(f"[SUCCESS] pz{args.patientid} unified.")

if __name__ == "__main__":
    main()