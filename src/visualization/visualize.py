import os
import argparse
import numpy as np

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    HAS_PYVISTA=True
except ImportError:
    HAS_PYVISTA=False

AT_RISK_THRESHOLDS={
    "TAWSS": ("lt", 0.4),
    "OSI": ("gt", 0.3),
    "ECAP": ("gt", 1.4),
    "RRT": ("gt", 5.0),
    "Pressure": ("gt", None),
}

COLORMAPS={
    "TAWSS": "Blues_r",
    "OSI": "hot",
    "ECAP": "RdYlGn_r",
    "RRT": "plasma",
    "Pressure": "viridis",
    "Velocity": "jet",
    "WSS": "coolwarm",
}

def visualize_field(vtp_path, field, out_dir):
    if not HAS_PYVISTA:
        raise ImportError("Pyvista is required for mesh visualization (pip install pyvista)")
    mesh=pv.read(vtp_path)
    patient_id=os.path.splitext(os.path.basename(vtp_path))[0]
    if field not in mesh.point_data:
        print(f"[WARN] Field {field} not found in {patient_id}")
        return
    arr=mesh.point_data[field]
    if arr.ndim==2:
        arr=np.linalg.norm(arr, axis=1)
        mesh.point_data[f"{field}_mag"]=arr
        plot_field=f"{field}_mag"
    else:
        plot_field=field

    cmap=COLORMAPS.get(field, "viridis")
    title=f"{patient_id} - {field}"

    plotter=pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, scalars=plot_field, cmap=cmap, show_scalar_bar=True, scalar_bar_args={"title": field})
    plotter.add_title(title, font_size=12)
    plotter.camera_position="xz"
    screenshot_path=os.path.join(out_dir, f"{patient_id}_{field}.png")
    plotter.screenshot(screenshot_path)
    plotter.close()
    print(f"Saved visualization for {field} to {screenshot_path}")

    if field in AT_RISK_THRESHOLDS:
        direction, threshold=AT_RISK_THRESHOLDS[field]
        if threshold is None:
            threshold=float(np.mean(arr)+2*np.std(arr)) 
        risk_mask=(arr<threshold) if direction=="lt" else (arr>threshold)
        mesh.point_data["at_risk"]=risk_mask.astype(np.float32)

        plotter2=pv.Plotter(off_screen=True)
        plotter2.add_mesh(mesh, scalars="at_risk", cmap="RdYlGn_r", clim=[0,1], show_scalar_bar=True, scalar_bar_args={"title": f"{field} - At Risk"})
        plotter2.add_title(f"{title} - At Risk Regions", font_size=12)
        plotter2.camera_position="xz"
        risk_path=os.path.join(out_dir, f"{patient_id}_{field}_at_risk.png")
        plotter2.screenshot(risk_path)
        plotter2.close()
        print(f"Saved at-risk visualization for {field} to {risk_path}")

def main():
    parser=argparse.ArgumentParser(description="Visualie CFD heatmaps")
    parser.add_argument("--vtp", required=True, help="Path to .vtp file")
    parser.add_argument("--field", default="TAWSS", choices=list(COLORMAPS.keys()), help="Field to visualize")
    parser.add_argument("--out", default="outputs/mesh_heatmaps/")
    parser.add_argument("--all_fields", action="store_true", help="render all fields")
    args=parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.all_fields:
        for f in COLORMAPS:
            visualize_field(args.vtp, f, args.out)
    else:
        visualize_field(args.vtp, args.field, args.out)
if __name__=="__main__":
    main()