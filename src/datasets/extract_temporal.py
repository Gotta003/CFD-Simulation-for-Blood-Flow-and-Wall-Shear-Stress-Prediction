from concurrent.futures import ProcessPoolExecutor, as_completed
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
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    
    if input_path.is_dir():
        subdirs=[d for d in input_path.iterdir() if d.is_dir() and d.name.endswith("-procs")]
        search_dir=input_path
        if subdirs:
            def get_proc_num(d):
                match=re.search(r'^(\d+)-procs', d.name)
                return int(match.group(1)) if match else 0
            target_proc_dir=sorted(subdirs, key=get_proc_num)[-1]
            print(f"[INFO] Auto-detected proc directory. Targeting: {target_proc_dir.name}")
            search_dir=target_proc_dir
        files=sorted(list(search_dir.rglob("*.vtp"))+list(search_dir.rglob("*.vtu")))
        if not files:
            raise FileNotFoundError(f"No .vtp/.vtu files found in {input_path}")
        entries=[]
        for p in files:
            match=re.search(r'(?:result_)?(\d+(?:\.\d+)?)', p.stem)
            if match:
                t=float(match.group(1))
            else:
                t=float(files.index(p))
            entries.append((t, p))
        entries.sort(key=lambda x: x[0])
        ts=[e[0] for e in entries]
        if ts and max(ts)==len(ts)-1:
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

def process_instant(path: Path, n_points: int, strategy: str, seed: int, ref_indices: np.ndarray | None=None, all_points: bool=False) -> dict | None:
    rng=np.random.default_rng(seed)
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
        np.isfinite(p_out) &
        np.isfinite(wx_out) & np.isfinite(wy_out) & np.isfinite(wz_out)
    )
    valid_idx=np.where(valid_mask)[0]
    if len(valid_idx)==0:
        print(f"[ERROR] No valid points in {path.name}")
        return None
    if all_points:
        ref_indices=np.arange(len(valid_idx))
        sel=valid_idx
    elif ref_indices is not None:
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
    parser.add_argument("--workers", type=int, default=os.cpu_count()-1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fixed_grid", action="store_true", help="Ise same spatial sample across all timesteps")
    parser.add_argument("--all_points", action="store_true", help="Use all points (overrides --n_points and --strategy)")

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
    if not entries:
        return
    ref_indices=None
    if args.fixed_grid and not args.all_points:
        print("Establishing fixed grid from first timestep...")
        first_t, first_path=entries[0]
        first_result=process_instant(first_path, args.n_points, args.strategy, args.seed, ref_indices=None, all_points=False)
        if first_result is None:
            print(f"[ERROR] Failed to process first timestep. Cannot establish fixed grid. Exiting...")
            return
        ref_indices=first_result["_ref_indices"]
        print(f"Fixed grid established with {len(ref_indices)} points.")
    result_dict={}
    rng=np.random.default_rng(args.seed)
    xyz, t, vx, vy, vz, p, wx, wy, wz=[], [], [], [], [], [], [], [], []
    mask=[]
    fields_found=set()

    print(f"\nLaunching parallel extraction with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx={
            executor.submit(process_instant, fpath, args.n_points, args.strategy, args.seed+i, ref_indices, args.all_points):
            i for i, (val, fpath) in enumerate(entries)
        }
        for future in as_completed(future_to_idx):
            i=future_to_idx[future]
            try:
                res=future.result()
                if res is not None:
                    result_dict[i]=res
                    fields_found.update(res["_fields_found"])
                    print(f"[DONE] [{i+1}/{len(entries)}] t={entries[i][0]:.4g} {entries[i][1].name} - fields: {res['_fields_found']}")
                else:
                    print(f"[FAILED] [{i+1}/{len(entries)}] t={entries[i][0]:.4g} {entries[i][1].name} - no valid points")
            except Exception as e:
                print(f"[ERROR] Failed to process {entries[i][1].name}: {e}")
        if not result_dict:
            print("[ERROR] No valid data extracted. Exiting...")
            return
        
    print(f"\nAssembling dataset from {len(result_dict)}")
    sorted_indices=sorted(result_dict.keys())
    for i in sorted_indices:
        result=result_dict[i]
        xyz.append(result["xyz"])
        t.append(entries[i][0])
        vx.append(result["vx"])
        vy.append(result["vy"])
        vz.append(result["vz"])
        p.append(result["p"]) 
        wx.append(result["wx"])
        wy.append(result["wy"])
        wz.append(result["wz"])
        mask.append(result["mask"])

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