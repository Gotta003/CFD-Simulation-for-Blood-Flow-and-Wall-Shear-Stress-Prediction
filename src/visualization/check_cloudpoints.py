import os
import re
import argparse
import numpy as np
from pathlib import Path 

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    HAS_PYVISTA=True
except ImportError:
    HAS_PYVISTA=False

COLORMAPS={
    "TAWSS": "Blues_r",
    "OSI": "hot",
    "RRT": "plasma"
}

def get_formatted_pid(name):
    digits=re.findall(r'\d+', name)
    if digits:
        return f"pz{int(digits[0]):03d}"
    return name

def check_sampled_points(npz_path, base_out_dir):
    if not HAS_PYVISTA:
        raise ImportError("pyvista is required for visualization.")
    data=np.load(npz_path)
    raw_name=Path(npz_path).stem
    pid=get_formatted_pid(raw_name)
    
    if "xyz" not in data:
        print(f"[ERROR] {pid} - 'xyz' not found in {npz_path}")
        return
    xyz=data["xyz"]
    fields=[k for k in data.keys() if k not in ["xyz", "mask"]]
    patient_out_dir=Path(base_out_dir) / pid
    patient_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Patient: {pid} (from {raw_name}.npz) ---")
    for field in fields:
        scalars=data[field]
        pc=pv.PolyData(xyz)
        pc[field]=scalars
        
        plotter=pv.Plotter(off_screen=True, window_size=(1200, 900))
        plotter.add_mesh(pc, scalars=field, cmap=COLORMAPS.get(field, "viridis"), point_size=8.0, render_points_as_spheres=True, show_scalar_bar=True, scalar_bar_args={"title": f"Sampled {field.upper()}"})
        plotter.add_title(f"NPZ Sanity Check - {pid} ({len(xyz)} points)", font_size=10)
        plotter.view_isometric()

        out_path=patient_out_dir / f"{field}.png"
        plotter.screenshot(out_path)
        plotter.close()
        print(f"--- Sanity Report: {pid} ---")
        print(f"Points: {len(xyz)}")
        print(f"{field} Range: [{scalars.min():.4f}, {scalars.max():.4f}]")
        print(f"Stored in: {out_path}")

def main():
    parser=argparse.ArgumentParser(description="Check .npz point cloud sanity")
    parser.add_argument("--input", required=True, help="Path to .npz point cloud")
    parser.add_argument("--out_dir", default="outputs/npz_checks/")
    args=parser.parse_args()
    input_path=Path(args.input)
    if input_path.is_dir():
        files=sorted(list(input_path.glob("*.npz")))
    else:
        files=[input_path]

    for f in files:
        check_sampled_points(str(f), args.out_dir)
    
if __name__=="__main__":
    main()