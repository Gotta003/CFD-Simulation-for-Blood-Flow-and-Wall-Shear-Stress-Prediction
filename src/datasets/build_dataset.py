import os
import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOKEN_MAP: dict[str, list[str]]={
    "none": [],
    "type i": ["endoleak_type1"],
    "type ia": ["endoleak_type1a"],
    "type ib": ["endoleak_type1b"],
    "type ii": ["endoleak_type2"],
    "type iii": ["endoleak_type3"],
    "type iv": ["endoleak_type4"],
    "graft migration": ["other_migration"],
    "thrombosis": ["other_thrombosis"],
    "reintervention": ["other_reintervention"],
    "rupture": ["other_rupture"]
}

ENDOLEAK_COLS=["endoleak_type1", "endoleak_type1a", "endoleak_type1b", "endoleak_type2", "endoleak_type3", "endoleak_type4"]
OTHER_COLS=["other_migration", "other_thrombosis", "other_reintervention", "other_rupture"]
TARGET_COLS=ENDOLEAK_COLS+OTHER_COLS
ALL_LABEL_COLS=TARGET_COLS+["any_endoleak"]

CFD_PREFIXES=["tawss", "osi", "ecap", "rrt", "pressure", "wss", "velocity", "vorticity", "divergence", "traction"]

def get_formatted_pid(name: str) -> str:
    name_str=str(name).strip().lower()
    digits=re.findall(r'\d+', name)
    if digits:
        return f"pz{int(digits[0]):03d}"
    return name_str

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map={c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        clear_cand=cand.lower().replace(" ", "_").strip()
        if clear_cand in lower_map:
            return lower_map[clear_cand]
    return None

def load_outcomes(path: str) -> pd.DataFrame:
    df=pd.read_csv(path)
    df.columns=[c.strip() for c in df.columns]
    id_col=_find_col(df, ["ID"])
    if id_col is None:
        raise ValueError(
            f"Cannot find a Patient ID column in {path}.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Expected one ID"
        )
    df=df.rename(columns={id_col:"patient_id"})
    df["patient_id"]=df["patient_id"].astype(str).str.strip()
    op_col=_find_col(df, ["Requires_Op"])
    if op_col is None:
        raise ValueError(
            f"Cannot find an operation flag column in {path}."
            f"Columns found: {list(df.columns)}"
        )
    df["operation_flag"]=(df[op_col].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0, '1': 1, '0': 0}).fillna(0).astype(int))
    if op_col!="operation_flag":
        df=df.drop(columns=[op_col])
    comp_col=_find_col(df, ["Complications"])
    if comp_col is None:
        print(f"[WARN] No complication type column found, then all type labels will be 0\nColumns found: {list(df.columns)}")
        df["complication_raw"]="none"
    else:
        df["complication_raw"]=df[comp_col].fillna("none").astype(str).str.strip()
        if comp_col!="complication_raw":
            df=df.drop(columns=[comp_col])
    return df
   
def _parse_cell(cell: str) -> tuple[dict[str, int], list[str]]:
    labels: dict[str, int]={col: 0 for col in ALL_LABEL_COLS}
    unknown: list[str]=[]
    raw=cell.strip().lower()
    if raw in (""):
        return labels, unknown
    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        if tok in TOKEN_MAP:
            for col in TOKEN_MAP[tok]:
                labels[col]=1
        else:
            unknown.append(tok)
    labels["any_endoleak"]=int(labels["endoleak_type1"] or labels["endoleak_type1a"] or labels["endoleak_type1b"] or labels["endoleak_type2"] or labels["endoleak_type3"] or labels["endoleak_type4"])
    return labels, unknown   
        
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    all_unknown: set[str]=set()
    rows: list[dict[str, int]]=[]
    for cell in df["complication_raw"]:
        label_dict, unknowns=_parse_cell(str(cell))
        all_unknown.update(unknowns)
        rows.append(label_dict)
    if all_unknown:
        print("\n[WARN] Not recognised complication token, needed addition to TOKEN_MAP:")
        for t in sorted(all_unknown):
            print(f"    '{t}'")
    label_df=pd.DataFrame(rows, index=df.index)
    return pd.concat([df, label_df], axis=1)

def validate_merge(merged: pd.DataFrame, n_feat: int, n_out: int) -> list[str]:
    issues=[]
    n=len(merged)
    if n==0:
        issues.append("Merge produced zero rows - check patient_id")
        return issues
    if n<n_feat*0.5:
        issues.append(f"Only {n}/{n_feat} feature patients matched outcomes ({n_feat-n} unmatched)")
    for col in ENDOLEAK_COLS:
        n_pos=int(merged[col].sum())
        pct=100*n_pos/n
        if n_pos==0:
            issues.append(f"Label '{col}' has zero positives")
        elif pct>70:
            issues.append(f"Label '{col}' is {pct:.0f}% positive")
    return issues

def get_cfd_feature_cols(df: pd.DataFrame) -> list[str]:
    return sorted(col for col in df.columns if any(col.lower().startswith(p) for p in CFD_PREFIXES) and pd.api.types.is_numeric_dtype(df[col]))

def write_label_summary(df: pd.DataFrame, out_path: str) -> None:
    n=len(df)
    if n==0:
        print(f"No data to summarize for label summary at {out_path}")
        return
    
    t1=df["endoleak_type1"].astype(bool)
    t1a=df["endoleak_type1a"].astype(bool)
    t1b=df["endoleak_type1b"].astype(bool)
    t2=df["endoleak_type2"].astype(bool)
    t3=df["endoleak_type3"].astype(bool)
    t4=df["endoleak_type4"].astype(bool)
    lines=["Label balance summary", "="*52, f"Total patients: {n}", "", "Operation flag", f"required: {int(df['operation_flag'].sum())}", f"({100*df['operation_flag'].mean():.1f}%)", f"none: {n-int(df['operation_flag'].sum())}", "", "Endoleak labels (multi-hot):",]
    for col in ENDOLEAK_COLS+["any_endoleak"]:
        n_pos=int(df[col].sum())
        lines.append(f"{col:22s}: {n_pos:4d} pos ({100*n_pos/n:5.1f}%) {n-n_pos:4d} neg ({100*(n-n_pos)/n:5.1f}%)")
    lines+=["","Co-occurence (patients with multiple endoleak types):",     
        f"Type I (Pure) only: {int((t1 & ~t1a & ~t1b & ~t2 & ~t3 & ~t4).sum())}",
        f"Type IA only:      {int((t1 &  t1a & ~t1b & ~t2 & ~t3 & ~t4).sum())}",
        f"Type IB only:      {int((t1 & ~t1a &  t1b & ~t2 & ~t3 & ~t4).sum())}",
        f"Type I+ (A&B) only: {int((t1 &  t1a &  t1b & ~t2 & ~t3 & ~t4).sum())}",
        f"Type II only:      {int((~t1 & ~t2 &  t2 & ~t3 & ~t4).sum())}",
        f"Type III only:     {int((~t1 & ~t2 & ~t3 &  t3 & ~t4).sum())}",
        f"Type IV only:      {int((~t1 & ~t2 & ~t3 & ~t4 &  t4).sum())}",
        f"Type I & II:   {int((t1 & t2 & ~t3 & ~t4).sum())}",
        f"Type I & III:  {int((t1 & t3 & ~t2 & ~t4).sum())}",
        f"Type I & IV:   {int((t1 & t4 & ~t2 & ~t3).sum())}",
        f"Type II & III: {int((~t1 & t2 & t3 & ~t4).sum())}",
        f"Type II & IV:  {int((~t1 & t2 & t4 & ~t3).sum())}",
        f"Type III & IV: {int((~t1 & t3 & t4 & ~t2).sum())}",
        "",
        "Other complications:"
    ]
    for col in OTHER_COLS:
        n_pos=int(df[col].sum())
        lines.append(f"{col:22s}: {n_pos:4d} ({100*n_pos/n:.1f}%)")
        lines+=["", "Raw complication cell frequencies (top 30):"]
        for val, cnt in df["complication_raw"].value_counts().head(30).items():
            lines.append(f" {str(val):45s}: {cnt}")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
    print(f"Label summary -> {out_path}")
        
def main():
    parser=argparse.ArgumentParser(description="Merge CFD features with clinical outcomes")
    parser.add_argument("--features", default="outputs/features/features.csv")
    parser.add_argument("--outcomes", default="data/labels/outcomes.csv")
    parser.add_argument("--out_dir", default="outputs/dataset/")
    parser.add_argument("--pointcloud_dir", default="data/pointclouds/", help="Dir with .npz files used only to add a has_npz flag")
    args=parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    #Features
    print(f"Loading features: {args.features}")
    feat_df=pd.read_csv(args.features)
    feat_df["patient_id"]=feat_df["patient_id"].apply(get_formatted_pid)
    print(f"{len(feat_df)} patients, {len(feat_df.columns)} columns")
    #Outcomes
    print(f"Loading outcomes: {args.outcomes}")
    out_df=load_outcomes(args.outcomes)
    out_df["patient_id"]=out_df["patient_id"].apply(get_formatted_pid)
    out_df=encode_labels(out_df)
    print(f"{len(feat_df)} patients, {len(feat_df.columns)} columns")
    for col in ENDOLEAK_COLS+["any_endoleak"]:
        print(f"    {col}: {int(out_df[col].sum())} positive")
    #Merge
    keep_out=["patient_id", "operation_flag", "complication_raw"]+ALL_LABEL_COLS
    merged=feat_df.merge(out_df[keep_out], on="patient_id", how="inner", validate="1:1")
    print(f"\nMerged: {len(merged)} patients ({len(feat_df)-len(merged)} unmatched and dropped)")
    if len(merged)==0:
        print("\n[CRITICAL ERROR] Merge resulted in 0 patients.")
        print(f"Sample feature IDs: {feat_df['patient_id'].head(3).tolist()}")
        print(f"Sample outcome IDs: {out_df['patient_id'].head(3).tolist()}")
        return
    #Point-Cloud Avail 
    if os.path.isdir(args.pointcloud_dir):
        npz_ids={Path(p).stem for p in Path(args.pointcloud_dir).glob("*.npz")}
        merged["has_npz"]=merged["patient_id"].isin(npz_ids).astype(int)
        print(f"Point clouds (.npz): {int(merged['has_npz'].sum())}/{len(merged)}")
    else:
        merged["has_npz"]=0
    #Validate
    issues=validate_merge(merged, len(feat_df), len(out_df))
    if issues:
        print("\n[VALIDATION ISSUES]")
        for i in issues:
            print(f"    ! {i}")
    else:
        print("Validation OK")
    #Adjust Missing CFD
    feature_cols=get_cfd_feature_cols(merged)
    n_miss=merged[feature_cols].isna().sum().sum()
    if n_miss>0:
        print(f"\nInputing {n_miss} missing CFD values with column median")
        for col in feature_cols:
            merged[col]=merged[col].fillna(merged[col].median())
    #Column Ordering
    meta=["patient_id", "operation_flag", "complication_raw", "has_npz"]
    others=[c for c in merged.columns if c not in meta + feature_cols]
    merged=merged[meta+TARGET_COLS+["any_endoleak"]+feature_cols+others]
    #Saving
    dataset_path=os.path.join(args.out_dir, "dataset.csv")
    merged.to_csv(dataset_path, index=False)
    print(f"\nDataset saved -> {dataset_path} {merged.shape}")
    feat_list_path=os.path.join(args.out_dir, "feature_columns.txt")
    Path(feat_list_path).write_text("\n".join(feature_cols))
    print(f"Feature list -> {feat_list_path} ({len(feature_cols)} columns)")
    write_label_summary(merged, os.path.join(args.out_dir, "label_summary.txt"))
    
if __name__=="__main__":
    main()