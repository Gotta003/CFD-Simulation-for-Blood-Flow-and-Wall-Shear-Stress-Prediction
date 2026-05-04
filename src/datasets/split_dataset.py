import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

TARGET_COLS=["endoleak_type1a", "endoleak_type1b", "endoleak_type2", "endoleak_type3", "endoleak_type4", "other_migration", "other_thrombosis", "other_reintervention", "other_rupture"]


def strat_key(df: pd.DataFrame) -> np.ndarray:
    t1=df["endoleak_type1"].values.astype(int)
    t2=df["endoleak_type2"].values.astype(int)
    t3=df["endoleak_type3"].values.astype(int)
    t4=df["endoleak_type4"].values.astype(int)
    return t1*8+t2*4+t3*2+t4

def stratified_split(ids: np.ndarray, strat: np.ndarray, test_pct: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    unique_strata=np.unique(strat)
    train_idx, test_idx=[], []
    for s in unique_strata:
        mask=(strat==s)
        s_ids=ids[mask]
        s_ids=rng.permutation(s_ids)
        n_test=max(1, int(round(len(s_ids)*test_pct)))
        n_test=min(n_test, len(s_ids)//2) if len(s_ids)>1 else 0
        test_idx.extend(s_ids[:n_test].tolist())
        train_idx.extend(s_ids[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)

def kfold_stratified(ids: np.ndarray, strat: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_bins: list[list]=[[] for _ in range(n_folds)]
    unique_strata=np.unique(strat)
    for s in unique_strata:
        mask=(strat==s)
        s_ids=rng.permutation(ids[mask])
        for i, pid in enumerate(s_ids):
            fold_bins[i%n_folds].append(pid)
    folds=[]
    for k in range(n_folds):
        val_ids=np.array(fold_bins[k])
        train_ids=np.concatenate([np.array(fold_bins[j]) for j in range(n_folds) if j!=k])
        folds.append((train_ids, val_ids))
    return folds

def split_summary(df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], test_ids: np.ndarray, out_path: str) -> None:
    lines=["Split summary", "="*50]
    n=len(df)
    lines.append(f"\nTest Set: {len(test_ids)} patients")
    
    def bal(ids):
        sub=df[df["patient_id"].isin(ids)]
        parts=[]
        for col in TARGET_COLS:
            pos=int(sub[col].sum())
            parts.append(f"{col.replace("endoleak_", "el_")}={pos}/{len(sub)} ({100*pos/max(len(sub), 1):.0f}%)")
        return "  ".join(parts)
    
    lines.append(f"{bal(test_ids)}")
    lines.append("")
    for k, (tr, va) in enumerate(folds):
        lines.append(f"Fold {k}:")
        lines.append(f" train {len(tr):4d} | {bal(tr)}")
        lines.append(f" val {len(va):4d} | {bal(va)}")
        lines.append("")
    lines.append("Recommended scale_pos_weight (neg/pos) per label:")
    for col in TARGET_COLS:
        n_pos=int(df[df["patient_id"].isin(np.concatenate([t for t,_ in folds]))][col].sum())
        n_neg=sum(len(t) for t,_ in folds)-n_pos
        w=n_neg/max(n_pos, 1)
        lines.append(f" {col:20s}: {w:.2f}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def main():
    parser=argparse.ArgumentParser(description="Stratified k-fold split for EVAR dataset")
    parser.add_argument("--dataset", default="outputs/dataset/dataset.csv")
    parser.add_argument("--out_dir", default="outputs/splits/")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--test_pct", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.dataset)
    df["patient_id"]=df["patient_id"].astype(str)
    ids=df["patient_id"].values
    strat=strat_key(df)    
    rng=np.random.default_rng(args.seed)
    #Test split
    trainval_ids, test_ids=stratified_split(ids, strat, args.test_pct, rng)
    print(f"Test set: {len(test_ids)} patients")
    print(f"Train and Val Set: {len(trainval_ids)} patients")
    #5 fold CV on train and val set
    tv_mask=df["patient_id"].isin(trainval_ids)
    tv_df=df[tv_mask].reset_index(drop=True)
    tv_ids=tv_df["patient_id"].values
    tv_strat=strat_key(tv_df)
    folds=kfold_stratified(tv_ids, tv_strat, args.n_folds, rng)

    test_path=os.path.join(args.out_dir, "test_ids.npy")
    np.save(test_path, test_ids)
    print(f"Saved: {test_path}")
    for k, (tr, va) in enumerate(folds):
        np.save(os.path.join(args.out_dir, f"fold_{k}_train_ids.npy"), tr)
        np.save(os.path.join(args.out_dir, f"fold_{k}_val_ids.npy"), va)
        n_tr, n_va=len(tr), len(va)
        print(f"Fold {k}: train={n_tr} val={n_va}")
    summary_path=os.path.join(args.out_dir, "split_summary.txt")
    split_summary(df, folds, test_ids, summary_path)
    print(f"\nSplit summary -> {summary_path}")
    
    print("\nLabel balance (train+val set):")
    for col in TARGET_COLS:
        n_pos=int(tv_df[col].sum())
        n_neg=len(tv_df)-n_pos
        print(f"    {col}: pos={n_pos} neg={n_neg} ratio={n_neg/max(n_pos,1):.1f}:1")

if __name__=="__main__":
    main()