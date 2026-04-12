import argparse
import os
import pandas as pd

FILE_NAME="./data/labels/outcomes.csv"
EMPTY_ROW={
    "ID": None,
    "Requires_Op": "No",
    "Complications": "",
    "Notes": "",
    "Segmentation": False,
    "Report_Analysis": False,
    "CFD_Simulations": False,
    "Image_Processing": False,
    "Labeling": False,
    "Examined_Files": ""
}

def main():
    parser=argparse.ArgumentParser(description="Add blank patient rows to outcomes.csv")
    parser.add_argument("--min", type=int, default=1, dest="id_min", help="First patient ID to add (default: 1)")
    parser.add_argument("--max", type=int, default=10, dest="id_max", help="Last patient ID to add (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be added without writing")
    args=parser.parse_args()
    
    if args.id_min>args.id_max:
        print(f"[ERROR] --min ({args.id_min}) must be <= --max ({args.id_max})")
        return
    
    if os.path.exists(FILE_NAME) and os.path.getsize(FILE_NAME):
        df=pd.read_csv(FILE_NAME).fillna("")
    else:
        print(f"Warning: {FILE_NAME} is empty or missing")
        df=pd.DataFrame(columns=list(EMPTY_ROW.keys()))
    existing_ids=set(df["ID"].astype(int).tolist()) if not df.empty else set()
    requested=set(range(args.id_min, args.id_max+1))
    to_add=sorted(requested-existing_ids)
    
    if not to_add:
        print(f"Existing Patients: {len(existing_ids)}")
        print(f"Requested range: {args.id_min}-{args.id_max} ({len(requested)} IDs)")
        print(f"To be added: {len(to_add)} patients")

    if args.dry_run:
        print("\n[Dry-Run] Would add IDs:", to_add)
        return
    
    new_rows=[{**EMPTY_ROW, "ID": pid} for pid in to_add]
    df=pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["ID"]=df["ID"].astype(int)
    df=df.sort_values("ID").reset_index(drop=True)
    
    os.makedirs(os.path.dirname("FILE_NAME") or ".", exist_ok=True)
    df.to_csv(FILE_NAME, index=False)
    print(f"\nAdded {len(to_add)} patients -> {FILE_NAME}")
    print(f"Total patients in file: {len(df)}")

if __name__=="__main__":
    main()
