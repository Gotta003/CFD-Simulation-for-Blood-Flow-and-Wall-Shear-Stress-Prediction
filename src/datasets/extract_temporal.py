import argparse
import json
import os
import re
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA=True
except ImportError:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_PYVISTA=False
    
FIELD_ALIASES: dict[str, list[str]]={
    "velocity": ["Velocity"],
    "pressure": ["Pressure"],
    "wss": ["WSS"],
}
    
PHYSICAL_BOUND: dict[str, tuple[float, float]]={
    "velocity": (0.0, 10.0),
    "pressure": (-500.0, 500.0),
    "wss": (0.0, 200.0)
}

def _load(path: str):
    if HAS_PYVISTA:
        return pv.read(path), "pv"
    ext=Path(path).suffix.lower()
    if ext==".vtp":
        r=vtk.vtkXMLPolyDataReader()
    elif ext==".vtu":
        r=vtk.vtkXMLUnstructuredGridReader()
    else:
        r=vtk.vtkXMLGenericDataObjectReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput(), "vtk"

def _points(mesh, backend: str) -> np.ndarray:
    if backend=="pv":
        return np.asarray(mesh.points, dtype=np.float32)
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(mesh.GetPoints().getData()).astype(np.float32)

def _array(mesh, name: str, backend: str) -> np.ndarray | None:
    if backend=="pv":
        for store in (mesh.point_data, mesh.cell_data):
            if name in store:
                arr=np.asarray(store[name], dtype=np.float64)
                return arr
        return None
    from vtk.util.numpy_support import vtk_to_numpy
    pd=mesh.GetPointData()
    raw=pd.GetArray(name)
    if raw is None:
        return None
    return vtk_to_numpy(raw).astype(np.float64)

def _get_field(mesh, field: str, backend: str) -> np.ndarray | None:
    for alias in FIELD_ALIASES[field]:
        arr=_array(mesh, alias, backend)
        if arr is not None:
            if arr.ndim==2:
                arr=np.linalg.norm(arr, axis=1)
            return arr.astype(np.float32)
    return None    

def discover_files(input_path):
    if input_path.is_dir():
        files=sorted(list(input_path.glob("*.vtp"))+list(input_path.glob("*.vtu")))
        if not files:
            raise FileNotFoundError(f"No .vtp/.vtu files found in {input_path}")
        entries=[]
        for p in files:
            nums=re.findall(r"(\d+(?:\.\d+)?)", p.stem)
            t=float(nums[-1]) if nums else float(files.index(p))
            entries.append((t,p))
        entries.sort(key=lambda x: x[0])
        #Normalization time to index
        ts=[e[0] for e in entries]
        if max(ts)==len(ts)-1:
            print("[INFO] No physical timestamps found - using step index as t.")
        return entries
    return [(0.0, input_path)]

def _get_velocity_components(mesh, backend: str) -> tuple[np.ndarray | None, ...]:
    for alias in FIELD_ALIASES["velocity"]:
        arr=_array(mesh, alias, backend)
        if arr is not None and arr.ndim==2 and arr.shape[1]>=3:
            return(
                arr[:,0].astype(np.float32),
                arr[:,1].astype(np.float32),
                arr[:,2].astype(np.float32)
            )
    return None, None, None

def _get_wss_components(mesh, backend: str) -> tuple[np.ndarray | None, ...]:
    for alias in FIELD_ALIASES["wss"]:
        arr=_array(mesh, alias, backend)
        if arr is not None and arr.ndim==2 and arr.shape[1]>=3:
            return(
                arr[:,0].astype(np.float32),
                arr[:,1].astype(np.float32),
                arr[:,2].astype(np.float32)
            )
    return None, None, None

def _clamp(arr: np.ndarray, field: str) -> np.ndarray:
    lo, hi=PHYSICAL_BOUND.get(field, (-np.inf, np.inf))
    return np.clip(arr, lo, hi)

def _random_sample(n_total: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    if n_sample>=n_total:
        return np.arange(n_total)
    return rng.choice(n_total, size=n_total, replace=False)

def _fps(points: np.ndarray, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    n=len(points)
    if n_sample>=n:
        return np.arange(n)
    sel=np.zeros(n_sample, dtype=np.int64)
    sel[0]=rng.integers(0,n)
    dists=np.full(n, np.inf)
    for i in range(1, n_sample):
        d=np.sum((points-points[sel[i-1]])**2, axis=1)
        dists=np.minimum(dists, d)
        sel[i]=int(np.argmax(dists))
    return sel

def process_instant(path: Path, n_points: int, strategy: str, rng: np.random.Generator, ref_indices: np.ndarray | None=None) -> dict | None:
    try:
        mesh, backend=_load(str(path))
    except Exception as e:
        print(f"[ERROR] Cannot load {path.name}: {e}")
        return None
    pts=_points(mesh, backend)
    n=len(pts)
    if n==0:
        print(f"[WARN] No points in {path.name}")
        return None
    #Extraction
    vx, vy, vz=_get_velocity_components(mesh, backend)
    wx, wy, wz=_get_wss_components(mesh, backend)
    p_arr=_get_field(mesh, "pressure", backend)
    found=[]
    zeros=np.zeros(n, dtype=np.float32)
    
    def _safe(arr, name):
        if arr is None:
            print(f"[WARN] Field {name} missing in {path.name} - filling zeros")
            return zeros.copy()
        found.append(name)
        return _clamp(arr, name if name in PHYSICAL_BOUND else "velocity")
    
    vx_out=_safe(vx, "vx")
    vy_out=_safe(vy, "vy")
    vz_out=_safe(vz, "vz")
    p_out=_safe(p_arr, "pressure")
    wx_out=_safe(wx, "wx")
    wy_out=_safe(wy, "wy")
    wz_out=_safe(wz, "wz")
    valid_mask=(
        np.all(np.isfinite(pts), axis=1) &
        np.isfinite(vx_out) & np.isfinite(vy_out) & np.isfinite(vz_out) &
        np.isfinite(p_out),
        np.isfinite(wx_out) & np.isfinite(wy_out) & np.isfinite(wz_out)
    )
    valid_idx=np.where(valid_mask)[0]
    if len(valid_idx)==0:
        print(f"[ERROR] No valid points in {path.name}")
        return None
    if ref_indices is not None:
        sel=valid_idx[np.clip(ref_indices, 0, len(valid_idx)-1)]
    else:
        pts_valid=pts[valid_idx]
        if strategy=="fps":
            local=_fps(pts_valid, n_points, rng)
        else:
            local=_random_sample(len(valid_idx), n_points, rng)
        if len(local)<n_points:
            pad=rng.choice(local, size=n_points-len(local), replace=True)
            local=np.concatenate([local, pad])
        ref_indices=local
        sel=valid_idx[local]
    return {
        "xyz": pts[sel],
        "vx": vx_out[sel],
        "vy": vy_out[sel],
        "vz": vz_out[sel],
        "p": p_out[sel],
        "wx": wx_out[sel],
        "wy": wy_out[sel],
        "wz": wz_out[sel],
        "mask": valid_mask[sel],
        "_ref_indices": ref_indices,
        "_fields_found": found
    }

def main():
    parser=argparse.ArgumentParser(description="Extraction of temporal point cloud (x, y, z, t, vx, vy, vz, px, py, pz, wss)")
    parser.add_argument("--input", required=True, help="Directory .vtp/.vtu or single .vtp/.vtu")
    parser.add_argument("--out_dir", default="data/temporal/")
    parser.add_argument("--name", default=None, help="Output filename stem (default: input stem)")
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--strategy", choices=["fps", "random"], default="fps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed_grid", action="store_true", help="Ise same spatial sample across all timesteps")

    args=parser.parse_args()
    input_path=Path(args.input)
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem=args.name or input_path.stem
    out_path=out_dir/f"{stem}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {out_path} already exists (use --overwrite)")
        return
    print(f"Discovering time steps in: {input_path}")
    entries=discover_files(input_path)
    print(f"Found {len(entries)} files")
    rng=np.random.default_rng(args.seed)
    ref_indices=None
    xyz=[]
    t=[]
    vx=[]
    vy=[]
    vz=[]
    wx=[]
    wy=[]
    wz=[]
    p=[]
    mask=[]
    fields_found: set[str]=set()
    for i, (t_val, fpath) in enumerate(entries):
        print(f" [{i+1}/{len(entries)}] t={t_val:.4g} {fpath.name}")
        result=process_instant(fpath, args.n_points, args.strategy, rng, ref_indices=ref_indices if args.fixed_grid else None)
        if result is None:
            print(f"[SKIP] Failed - skipping this timestep")
            continue
        if args.fixed_grid and ref_indices is None:
            ref_indices=result["_ref_indices"]
        xyz.append(result["xyz"])
        t.append(t_val)
        vx.append(result["vx"])
        vy.append(result["vy"])
        vz.append(result["vz"])
        p.append(result["p"]) 
        wx.append(result["wx"])
        wy.append(result["wy"])
        wz.append(result["wz"])
        mask.append(result["mask"])
        fields_found.update(result["_ref_indices"])
    if not t:
        print("[ERROR] No shots processed. Exiting...")
        return
    T=len(t)
    N=args.n_points
    print(f"\nAssembling dataset: T={T} timesteps, N={N} points/step")
    data={
        "xyz": np.stack(xyz, axis=0),
        "t": np.stack(t, dtype=np.float32),
        "vx": np.stack(vx, axis=0),
        "vy": np.stack(vy, axis=0),
        "vz": np.stack(vz, axis=0),
        "p": np.stack(p, axis=0),
        "wx": np.stack(wx, axis=0),
        "wy": np.stack(wy, axis=0),
        "wz": np.stack(wz, axis=0),
        "mask": np.stack(mask, axis=0),
        "fields_found": np.array(sorted(fields_found)),
    }
    np.savez_compressed(str(out_path), **data)
    print(f"\nSaved in {out_path}")
    print(f"    Shapes: xyz{data['xyz'].shape} t{data['t'].shape}")
    print(f"    Fields found in files: {sorted(fields_found)}")
    print(f"    t range: [{data['t'].min():.4g}, {data['t'].max():.4g}]")
 
if __name__=="__main__":
    main()