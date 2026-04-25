import argparse
import json
import os
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    HAS_PYVISTA=True
except ImportError:
    HAS_PYVISTA=False

try:
    import imageio
    HAS_IMAGEIO=True
except ImportError:
    HAS_IMAGEIO=False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL=True
except ImportError:
    HAS_MPL=False

CMAPS={
    "vx": "RdBu",
    "vy": "RdBu",
    "vz": "RdBu",
    "p": "coolwarm",
    "wx": "plasma",
    "wy": "plasma",
    "wz": "plasma",
    "v_mag": "viridis",
}

LABELS={
    "vx": "Velocity X (m/s)",
    "vy": "Velocity Y (m/s)",
    "vz": "Velocity Z (m/s)",
    "p": "Pressure (Pa)",
    "wx": "Vorticity X (1/s)",
    "wy": "Vorticity Y (1/s)",
    "wz": "Vorticity Z (1/s)",
    "v_mag": "Velocity Magnitude (m/s)",
}

def load_npz(path: str) -> dict:
    raw=np.load(path, allow_pickle=True)
    data={k: raw[k] for k in raw.files}
    if all(f in data for f in {"vx", "vy", "vz"}):
        data["v_mag"]=np.sqrt(data["vx"]**2 + data["vy"]**2 + data["vz"]**2)
    return data

def _build_cloud(xyz: np.ndarray, scalars: dict[str, np.ndarray]) -> "pv.PolyData":
    cloud=pv.PolyData(xyz)
    for k, v in scalars.items():
        cloud[k]=v
    return cloud

def _render_field(plotter: "pv.Plotter", cloud: "pv.PolyData", field: str, clim: tuple[float, float] | None=None):
    clim=clim or (float(cloud[field].min()), float(cloud[field].max()))
    plotter.add_mesh(cloud, scalars=field, cmap=CMAPS.get(field, "viridis"), clim=clim, point_size=6.0, render_points_as_spheres=True, show_scalar_bar=True, scalar_bar_args={"title": LABELS.get(field, field), "n_labels": 5})

def mode_compare(data: dict, t_idx: int, out_dir: Path, stem: str):
    if not HAS_PYVISTA:
        raise ImportError("Pyvista required for --mode")
    fields=[f for f in ("vx", "vy", "vz", "p", "wx", "wy", "wz", "v_mag") if f in data]
    n_fields=len(fields)
    t_val=float(data["t"][t_idx])
    xyz=data["xyz"][t_idx]
    clims={}
    for f in fields:
        arr=data[f]
        clims[f]=(float(arr.min()), float(arr.max()))
    scalars={f: data[f][t_idx] for f in fields}
    cloud=_build_cloud(xyz, scalars)
    nrows=2
    ncols=(n_fields + 1)//2
    plotter=pv.Plotter(shape=(nrows, ncols), off_screen=True, window_size=(500*ncols, 500*nrows))
    for i, f in enumerate(fields):
        plotter.subplot(i//ncols, i%ncols)
        _render_field(plotter, cloud, f, clim=clims[f])
        plotter.add_title(f"{LABELS.get(f, f)} | t={t_val:.4g}", font_size=9)
        plotter.view_isometric()
    out_path=out_dir/f"{stem}_compare_t{t_idx:03d}.png"
    plotter.screenshot(out_path)
    print(f"Compare image -> {out_path}")

def mode_animate(data: dict, field: str, out_dir: Path, stem: str, fps: int=5):
    if not HAS_PYVISTA:
        raise ImportError("Pyvista required for --mode animate")
    T=len(data["t"])
    frame_dir=out_dir/f"{stem}_frames_{field}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    arr_all=data[field]
    clim=(float(arr_all.min()), float(arr_all.max()))
    frames=[]
    for t_idx in range(T):
        t_val=float(data["t"][t_idx])
        xyz=data["xyz"][t_idx]
        cloud=_build_cloud(xyz, {field: arr_all[t_idx]})
        pl=pv.Plotter(off_screen=True, window_size=(1000, 800))
        _render_field(pl, cloud, field, clim=clim)
        pl.add_title(f"{stem} | {LABELS.get(field, field)} | t={t_val:.4g}", font_size=10)
        pl.view_isometric()
        out_frame=frame_dir/f"frame_{t_idx:04d}.png"
        pl.screenshot(str(out_frame))
        pl.close()
        frames.append(str(out_frame))
        print(f"  frame {t_idx+1}/{T}")
    if HAS_IMAGEIO:
        gif_path=out_dir/f"{stem}_{field}_animation.gif"
        with imageio.get_writer(gif_path, mode="I", duration=1.0/fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        print(f"Animation GIF -> {gif_path}")
    else:
        print(f"[INFO] imageio not installed - frames saved in {frame_dir}")

def mode_report(data: dict, out_dir: Path, stem: str):
    fields=[f for f in ("vx", "vy", "vz", "p", "wx", "wy", "wz", "v_mag") if f in data]
    t_vals=data["t"]
    T=len(t_vals)
    print(f"\n{'='*60}")
    print(f"  Temporal report – {stem}")
    print(f"  T={T} snapshots,  N={data['xyz'].shape[1]} points")
    print(f"  t ∈ [{t_vals.min():.4g}, {t_vals.max():.4g}]")
    print(f"{'='*60}")
    header = f"  {'Field':<10} {'t-mean min':>14} {'t-mean max':>14} "
    header += f"{'global min':>14} {'global max':>14}"
    print(header)
    print("  " + "-"*58)

    if HAS_MPL:
        fig, axes = plt.subplots(len(fields), 1, figsize=(9, 2.5 * len(fields)), sharex=True)
        if len(fields) == 1:
            axes = [axes]
        fig.suptitle(f"Field evolution over time – {stem}", fontsize=12)

    for i, field in enumerate(fields):
        arr = data[field]             
        mean_t = arr.mean(axis=1)        
        std_t  = arr.std(axis=1)

        print(f"  {field:<10} {mean_t.min():>14.4g} {mean_t.max():>14.4g} "
              f"{arr.min():>14.4g} {arr.max():>14.4g}")

        if HAS_MPL:
            ax = axes[i]
            ax.plot(t_vals, mean_t, lw=1.5, label="mean")
            ax.fill_between(t_vals,
                            mean_t - std_t,
                            mean_t + std_t,
                            alpha=0.2, label="±1σ")
            ax.set_ylabel(LABELS.get(field, field), fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    print(f"{'='*60}\n")

    if HAS_MPL:
        axes[-1].set_xlabel("time", fontsize=10)
        fig.tight_layout()
        out_path = out_dir / f"{stem}_report.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved report plot → {out_path}")
    else:
        print("[INFO] matplotlib not installed - skipping plot")

def main():
    parser=argparse.ArgumentParser(description="Visualize and analyze temporal evolution")
    parser.add_argument("--input", required=True, help="Path to .npz file")
    parser.add_argument("--out_dir", default="outputs/npz_checks/")
    parser.add_argument("--mode", choices=["compare", "animate", "report"], default="animate")
    parser.add_argument("--field", choices=["vx", "vy", "vz", "p", "wx", "wy", "wz", "v_mag"], default="v_mag", help="Field to visualize for animate mode")
    parser.add_argument("--t_idx", type=int, default=2, help="Time index to visualize in --mode compare")
    parser.add_argument("--fps", type=int, default=5, help="FPS for animation mode")
    args=parser.parse_args()
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path=Path(args.input)
    stem=npz_path.stem
    print(f"Loading data from {npz_path}...")
    data=load_npz(str(npz_path))
    T=len(data["t"])
    N=data["xyz"].shape[1]
    print(f"T={T} timestamps, N={N} points, t in [{data['t'].min():.4g}, {data['t'].max():.4g}]")
    if args.mode=="compare":
        mode_compare(data, args.t_idx, out_dir, stem)
    elif args.mode=="animate":
        mode_animate(data, args.field, out_dir, stem, fps=args.fps)
    elif args.mode=="report":
        mode_report(data, out_dir, stem)

if __name__=="__main__":
    main()