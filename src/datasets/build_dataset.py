import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COMPLICATION_MAP: dict[str, dict[str, int]]={}

TARGET_COLS=[]

CFD_PREFIXES=[]

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map={c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None

def load_outcomes(path: str) -> pd.DataFrame:
    df=pd.read_csv(path)
    df.columns=[c.strip() for c in df.columns]
    id_col=_find_col(df, ["patient_id", "patientid", "id", "patient"])
    if id_col is None:
        raise ValueError(
            f"Cannot find a Patient ID column in {path}.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Expected one of: patient_id, Patient_ID, id, patient"
        )
    df=df.rename(columns={id_col:"patient_id"})
    df["patient_id"]=df["patient_id"].astype(str).str.strip()
    op_col=_find_col(df, ["operation_flag", "operation", "reoperation", "reintervention", "requires_operation", "op_flag"])
    if op_col is None:
        raise ValueError(
            f"Cannot find an operation flag column in {path}."
            f"Columns found: {list(df.columns)}"
        )
    df["operation_flag"]=pd.to_numeric(df[op_col], errors="coerce").fillna(0).astype(int)
    if op_col!="operation_flag":
        df=df.drop(columns=[op_col])
    comp_col=_find_col(df, ["complication_type", "complication", "endoleak_type", "endoleak", "complication_label"])
    if comp_col is None:
        print(f"[WARN] No complication type column found, then all type labels will be 0\nColumns found: {list(df.columns)}")
        df["complication_type"]="none"
    else:
        df["complication_type"]=df[comp_col].fillna("none").astype(str).str.strip()
        if comp_col!="complication_type":
            df=df.drop(columns=[comp_col])
    return df
        
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    unknown=set()
    for _,row in df.iterrows():
        raw=row["complication_type"].lower().strip()
        mapping=COMPLICATION_MAP.get(raw)
        if mapping is None:
            unknown.add(raw)
            mapping={"endoleak_type1": 0, "endoleak_type2": 0, "endoleak_type3": 0}
            rows.append(mapping)
        if unknown:
            print(f"[WARN] Unrecognised complication strings:\n{sorted(unknown)}\nAdd them to COMPLICATION_MAP in build_dataset.py if needed.")
        label_df=pd.DataFrame(rows, index=df.index)
        return pd.concat([df, label_df], axis=1)
    
def main():
    parser=argparse.ArgumentParser(description="Merge CFD features with clinical outcomes")
    parser.add_argument("--features", default="outputs/features/features.csv")
    parser.add_argument("--outcomes", default="data/labels/outcomes.csv")
    parser.add_argument("out_dir", default="outputs/dataset/")
    parser.add_argument("--pointcloud_dir", default="data/pointclouds/", help="Dir with .npz files used only to add a has_npz flag")
    args=parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    #Features
    print(f"Loading features: {args.features}")
    feat_df=pd.read_csv(args.features)
    feat_df["patient_id"]=feat_df["patient_id"].astype(str).str.strip()
    print(f"{len(feat_df)} patients, {len(feat_df.columns)} columns")
    #Outcomes
    print(f"Loading outcomes: {args.outcomes}")
    out_df=load_outcomes(args.outcomes)
    out_df=encode_labels(out_df)
    print(f"{len(feat_df)} patients, {len(feat_df.columns)} columns")
if __name__=="__main__":
    main()