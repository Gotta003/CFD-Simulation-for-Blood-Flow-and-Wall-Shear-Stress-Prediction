import re
import csv
import argparse
from pathlib import Path

def extract_metrics(log_path: str) -> list[dict]:
    records=[]
    current_fold=None
    epoch_counter={}
    value_re=re.compile(r":\s*([\d.eE+\-]+)")
    with open(log_path, "r") as f:
        lines=f.readlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        fold_match=re.search(r"training fold (\d+)", line)
        if fold_match:
            current_fold=int(fold_match.group(1))
            epoch_counter[current_fold]=0
            i+=1
            continue
        if "val_loss" in line and current_fold is not None:
            try:
                val_loss=float(value_re.search(line).group(1))
                val_acc=float(value_re.search(lines[i+1]).group(1))
                val_auc=float(value_re.search(lines[i+2]).group(1))
            except (AttributeError, IndexError, ValueError) as e:
                print(f"[WARN] Could not parse metrics at line {i+1}: {e}")
                i+=1
                continue
            records.append({
                "fold": current_fold,
                "epoch": epoch_counter[current_fold],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            })
            epoch_counter[current_fold]+=1
            i+=3
            continue
        i+=1
    return records
    
def save_csv(records: list[dict], output_path: str) -> None:
    fieldnames=["fold", "epoch", "val_loss", "val_acc", "val_auc"]
    with open(output_path, "w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to {output_path}")
    
def print_summary(records: list[dict]) -> None:
    from collections import defaultdict
    folds=defaultdict(list)
    for r in records:
        folds[r["fold"]].append(r)
    print(f"\n{'Fold':<6} {'Epochs':<8} {'Best val_loss':<15} {'Best val_acc':<14} {'Best val_auc'}")
    print("-"*60)
    for fold in sorted(folds):
        fold_records=folds[fold]
        best_loss=min(r["val_loss"] for r in fold_records)
        best_acc=max(r["val_acc"] for r in fold_records)
        best_auc=max(r["val_auc"] for r in fold_records)
        print(f"{fold:<6} {len(fold_records):<8} {best_loss:<15.6f} {best_acc:<14.6f} {best_auc:.6f}")
    print(f"\nTotal records: {len(records)}")
        
def main():
    parser=argparse.ArgumentParser(description="Extract training metrics from a fold log file")
    parser.add_argument("--log_file", help="Path to training log file")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args=parser.parse_args()
    
    log_path=args.log_file
    output_path=args.output or str(Path(log_path).stem+"_metrics.csv")
    print(f"Reading: {log_path}")
    records=extract_metrics(log_path)
    print_summary(records)
    save_csv(records, output_path)
    
if __name__=="__main__":
    main()
